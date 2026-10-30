"""Extract frozen Prithvi WxC representations for every timestamp in our
regional 2024 MERRA-2 cube.

Embedding extraction point (see docs/prithvi_embedding_candidates.md for the
full derivation): PrithviWxC.forward() calls `x_encoded = self.encoder(unmasked)`
BEFORE the decoder. With `mask_ratio_inputs=0.0` nothing is masked, so
`unmasked` covers every token and `x_encoded` has shape
`[batch, n_global_mu, n_local_mu, embed_dim]` -- a full, spatially-structured
per-mask-unit token grid, not a single pooled vector. We capture this via a
forward hook on `model.encoder` (so we don't have to reimplement forward()),
run the model as normal (decoder output is computed but discarded -- it's
needed structurally since patch_embedding_static's weights were trained
under residual="climate" and require the climate tensor either way), and
save two pooled views per timestamp:

  - global_emb_*    : mean over ALL tokens (n_global_mu x n_local_mu) ->
                       one embed_dim-length vector per timestamp. Maximally
                       compact; loses all regional structure.
  - region{i}_emb_*  : mean over local tokens only, keeping one vector PER
                       MASK UNIT (n_global_mu of them). For the default 2x2
                       mask-unit crop that's 4 regions, each corresponding to
                       one quadrant of the regional box (see
                       region_centers() below for their approximate lat/lon
                       centers) -- a "modest spatially pooled representation"
                       that falls directly out of the model's own token
                       structure, no extra pooling logic invented.

`input_time`/`lead_time` are held FIXED across every timestamp (default -6h
/ 0h, i.e. "0-hour-ahead" nowcast mode, an explicitly supported pretraining
regime per the model card) so embeddings are comparable across time and only
ever depend on data up to and including the target timestamp -- no future
leakage into a downstream demand model.
"""
import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from PrithviWxC.dataloaders.merra2 import (
    Merra2Dataset,
    input_scalers,
    output_scalers,
    preproc,
    static_input_scalers,
)
from PrithviWxC.model import PrithviWxC

SURFACE_VARS = [
    "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP", "PS",
    "QV2M", "SLP", "SWGNT", "SWTNT", "T2M", "TQI", "TQL", "TQV", "TS",
    "U10M", "V10M", "Z0M",
]
STATIC_SURFACE_VARS = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
VERTICAL_VARS = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
LEVELS = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0]
NO_PADDING = {"level": [0, 0], "lat": [0, 0], "lon": [0, 0]}
# Extents for --pooled (lat_min, lat_max, lon_min, lon_max). "ne" is the default --token-box (New England);
# "northeast" an intermediate box; "wide" the --wide-box used for regional weather; the last five are the
# regional-weather areas of local/regional_weather.py.
POOL_EXTENTS = {
    "ne": (40.0, 48.0, -75.0, -66.25),
    "northeast": (38.0, 50.0, -80.0, -62.0),
    "wide": (32.0, 50.0, -90.0, -64.0),
    "NY-PA": (40.0, 45.0, -80.0, -75.0),
    "MidAtlantic": (36.0, 40.0, -80.0, -74.0),
    "GreatLakes": (38.0, 46.0, -90.0, -80.0),
    "Southeast": (32.0, 36.0, -90.0, -75.0),
    "Quebec-Mar": (45.0, 50.0, -80.0, -64.0),
}
# ERCOT (Texas) extents, for the ERCOT price replication (--pool-set ercot). Hub areas follow ERCOT's
# hubs: North (Dallas-Fort Worth, North Central), Houston, South (Austin, San Antonio, Corpus Christi, the Valley),
# West (Permian Basin, Abilene) and the Panhandle; "ercot" is the whole ERCOT footprint and "texas_wide" its
# surroundings (Oklahoma, New Mexico, Louisiana, northern Mexico) for a later regional test.
POOL_EXTENTS_ERCOT = {
    "ercot": (26.0, 36.5, -106.5, -93.5),
    "texas_wide": (24.0, 40.0, -110.0, -88.0),
    "north": (31.5, 34.5, -99.0, -95.0),
    "houston": (28.5, 31.0, -96.5, -94.0),
    "south": (26.0, 31.0, -100.5, -96.5),
    "west": (30.0, 34.0, -104.5, -99.0),
    "panhandle": (34.0, 36.5, -103.0, -100.0),
}
POOL_SETS = {"ne": POOL_EXTENTS, "ercot": POOL_EXTENTS_ERCOT}
# The global MERRA-2 grid has 361 latitude rows (both poles); the model uses 360.
# Dropping the last row is what the repo's own notebooks do.
GLOBAL_PADDING = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}


def region_centers(lat_min: float, lon_min: float, mask_unit_size_px, n_global_lat: int, n_global_lon: int):
    """Approximate lat/lon center of each mask unit, in the model's global
    token order (global_lat outer, global_lon inner -- see
    PrithviWxC.to_patching in model.py)."""
    lat_step, lon_step = 0.5, 0.625
    unit_lat_deg = mask_unit_size_px[0] * lat_step
    unit_lon_deg = mask_unit_size_px[1] * lon_step
    centers = []
    for gi in range(n_global_lat):
        for gj in range(n_global_lon):
            lat_c = lat_min + (gi + 0.5) * unit_lat_deg
            lon_c = lon_min + (gj + 0.5) * unit_lon_deg
            centers.append((lat_c, lon_c))
    return centers


def patch_grid(x_encoded: torch.Tensor, g_lat: int, g_lon: int, l_lat: int, l_lon: int) -> torch.Tensor:
    """Undo PrithviWxC.to_patching: [batch, g_lat*g_lon, l_lat*l_lon, C] ->
    [batch, lat_patches, lon_patches, C], so tokens can be indexed by
    position. Valid for the encoder output: the encoder's window shift is
    applied an even number of times, so tokens leave it in their original
    places."""
    b, _, _, c = x_encoded.shape
    x = x_encoded.reshape(b, g_lat, g_lon, l_lat, l_lon, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, g_lat * l_lat, g_lon * l_lon, c)


def check_normalised_inputs(x: torch.Tensor, model, n_levels: int) -> None:
    """Print how far the inputs sit from the model's own normalisation
    statistics, and stop if they are wildly off.

    Correctly prepared inputs are within a few standard deviations of the
    training mean. Finite but mis-assigned data (wrong units, wrong level
    order, wrong variable) shows up here as values tens to thousands of
    standard deviations out -- the first 2024 run had vertical profiles
    stored upside down and produced embeddings in the hundreds of millions.
    """
    z = (x - model.input_scalers_mu) / (model.input_scalers_sigma + model.input_scalers_epsilon)
    mean_abs = z.abs().mean(dim=(0, 1, 3, 4)).cpu().numpy()  # one value per channel
    names = list(SURFACE_VARS) + [f"{v}@{int(l)}" for v in VERTICAL_VARS for l in LEVELS]
    print("Normalised input check (mean |z| per variable; a few at most is expected):")
    for i, v in enumerate(SURFACE_VARS):
        print(f"  {v:9s} {mean_abs[i]:8.2f}")
    for j, v in enumerate(VERTICAL_VARS):
        block = mean_abs[len(SURFACE_VARS) + j * n_levels: len(SURFACE_VARS) + (j + 1) * n_levels]
        print(f"  {v:9s} " + " ".join(f"{b:7.2f}" for b in block) + f"   (levels {int(LEVELS[0])}..{int(LEVELS[-1])})")
    worst = int(mean_abs.argmax())
    if mean_abs[worst] > 20:
        raise SystemExit(
            f"Input '{names[worst]}' is on average {mean_abs[worst]:.0f} standard deviations from the model's "
            "training mean. The data is finite but not what the model expects (check units, level order, "
            "and variable mapping in the daily files). Stopping before extracting embeddings."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True, help="Downloaded prithvi.wxc.2300m.v1 snapshot (has config.yaml, *.pt, climatology/)")
    parser.add_argument("--merra2-surface-dir", type=Path, required=True)
    parser.add_argument("--merra2-vertical-dir", type=Path, required=True)
    parser.add_argument("--climatology-surface-dir", type=Path, required=True, help="Regionally-cropped climatology (from crop_climatology.py)")
    parser.add_argument("--climatology-vertical-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--input-time", type=int, default=-6)
    parser.add_argument("--lead-time", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-parquet", type=Path, required=True)
    parser.add_argument("--out-tokens", type=Path, default=None,
                        help="Also save the un-pooled patch tokens inside --token-box to this .npy (float16), with a .json sidecar")
    parser.add_argument("--layers", nargs="*", default=[],
                        help="Also save patch tokens from inside the encoder: 'embed' (before any attention) and/or block "
                             "indices that are multiples of 4 (0 = after the first local-attention block, before any "
                             "global mixing; the final block is always saved). Each adds a file the size of --out-tokens.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data-loading worker processes")
    parser.add_argument("--pool-box", type=float, nargs=4, default=[30.0, 60.0, -90.0, -50.0],
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="global_emb_* is the mean of the patch tokens in this box (default: the 30-60N, 90-50W "
                             "area that earlier regional runs pooled over)")
    parser.add_argument("--wide-box", type=float, nargs=4, default=None, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Also save the decoder output and the true inputs for T2M, QV2M, U10M, V10M, SWGNT over this larger "
                             "box, at full resolution (*_wide_recon5.npy, *_wide_inputs5.npy); for regional weather features")
    parser.add_argument("--token-box", type=float, nargs=4, default=[40.0, 48.0, -75.0, -66.25],
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Patches whose centres fall in this box are saved (default: New England)")
    parser.add_argument("--pooled", action="store_true",
                        help="Also save, per extent in POOL_EXTENTS: the final-layer tokens averaged over the extent's patches "
                             "(*_pooled_tokens.npy, [time, extent, embed_dim], float16), and all 160 input channels at issue time "
                             "and decoder output channels for the target time averaged over its pixels (*_pooled_inputs.npy, "
                             "*_pooled_recon.npy, [time, extent, channel], float32, normalised). Small; for the mechanistic "
                             "controls (wider latent state, upper-air conditions). Requires --out-tokens.")
    parser.add_argument("--pool-set", choices=sorted(POOL_SETS), default="ne",
                        help="Which extents --pooled averages over: 'ne' (New England and surroundings, default) or 'ercot'")
    parser.add_argument("--save-wide-patch-tokens", action="store_true",
                        help="Also save every final-layer patch token over the 'wide' extent (*_wide_tokens.npy, float16, "
                             "about 5-6 GB per year); for later interpretability, not needed for the pooled tests")
    args = parser.parse_args()
    extents = POOL_SETS[args.pool_set]
    if (args.pooled or args.save_wide_patch_tokens) and args.out_tokens is None:
        raise SystemExit("--pooled / --save-wide-patch-tokens need --out-tokens (they share its file stem).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Merra2Dataset.valid_timestamps globs MERRA_pres_*.nc from the SURFACE
    # path (upstream bug), so both file types must share one directory or it
    # finds no valid timestamps and dies with a bare StopIteration.
    if args.merra2_surface_dir.resolve() != args.merra2_vertical_dir.resolve():
        raise SystemExit(
            "--merra2-surface-dir and --merra2-vertical-dir must be the same directory: "
            "Merra2Dataset looks for the vertical files in the surface directory."
        )
    n_sfc = len(list(args.merra2_surface_dir.glob("MERRA2_sfc_????????.nc")))
    n_pres = len(list(args.merra2_surface_dir.glob("MERRA_pres_????????.nc")))
    print(f"Found {n_sfc} surface and {n_pres} vertical daily files in {args.merra2_surface_dir}")
    if n_sfc == 0 or n_pres == 0:
        raise SystemExit("No daily surface and/or vertical files found -- nothing to extract.")

    dataset = Merra2Dataset(
        time_range=(args.start, args.end),
        lead_times=[args.lead_time],
        input_times=[args.input_time],
        data_path_surface=args.merra2_surface_dir,
        data_path_vertical=args.merra2_vertical_dir,
        climatology_path_surface=args.climatology_surface_dir,
        climatology_path_vertical=args.climatology_vertical_dir,
        surface_vars=SURFACE_VARS,
        static_surface_vars=STATIC_SURFACE_VARS,
        vertical_vars=VERTICAL_VARS,
        levels=LEVELS,
        positional_encoding="fourier",
    )
    if len(dataset) == 0:
        raise SystemExit("No valid samples -- check that surface, vertical, AND climatology data all cover the requested range.")
    print(f"Dataset: {len(dataset)} timestamps")

    # One (timestamp, input_time, lead_time) tuple per index, in dataset order.
    # Only valid with shuffle=False below -- batch i must correspond to
    # timestamps[i*batch_size : (i+1)*batch_size].
    timestamps = [s[0][0] for s in dataset.samples]

    # Global input (361 x 576) is the setup the model works properly in: on
    # the official sample its output skill against climatology is 0.99 at
    # lead 0 and 0.95 at 6 h, against 0.82 and 0.44 inside New England when
    # it is only given a 60 x 64 pixel regional crop (sample_validation.py).
    padding = GLOBAL_PADDING if len(dataset.lats) == 361 else NO_PADDING
    n_lats_px = len(dataset.lats) + padding["lat"][0] + padding["lat"][1]
    n_lons_px = len(dataset.lons)
    grid_lats = np.asarray(dataset.lats, dtype=np.float64)[:n_lats_px]
    grid_lons = np.asarray(dataset.lons, dtype=np.float64)
    print(f"Model grid: {n_lats_px} x {n_lons_px} pixels ({'global' if padding is GLOBAL_PADDING else 'regional crop'}), "
          f"lat {grid_lats[0]}..{grid_lats[-1]}, lon {grid_lons[0]}..{grid_lons[-1]}")

    config = yaml.safe_load((args.weights_dir / "config.yaml").read_text())
    p = config["params"]
    mask_unit_size_px = p["mask_unit_size_px"]
    if n_lats_px % mask_unit_size_px[0] != 0 or n_lons_px % mask_unit_size_px[1] != 0:
        raise SystemExit(
            f"Grid {n_lats_px}x{n_lons_px} is not divisible by "
            f"mask_unit_size_px {mask_unit_size_px} -- re-run src/merra2/define_region.py."
        )
    n_global_lat = n_lats_px // mask_unit_size_px[0]
    n_global_lon = n_lons_px // mask_unit_size_px[1]
    n_global_mu = n_global_lat * n_global_lon
    embed_dim = p["embed_dim"]
    print(f"global_shape_mu = ({n_global_lat}, {n_global_lon}) -> {n_global_mu} regions, embed_dim={embed_dim}")

    clim_dir = args.weights_dir / "climatology"
    in_mu, in_sig = input_scalers(SURFACE_VARS, VERTICAL_VARS, LEVELS, clim_dir / "musigma_surface.nc", clim_dir / "musigma_vertical.nc")
    out_sig = output_scalers(SURFACE_VARS, VERTICAL_VARS, LEVELS, clim_dir / "anomaly_variance_surface.nc", clim_dir / "anomaly_variance_vertical.nc")
    static_mu, static_sig = static_input_scalers(clim_dir / "musigma_surface.nc", STATIC_SURFACE_VARS)

    model = PrithviWxC(
        in_channels=p["in_channels"],
        input_size_time=p["input_size_time"],
        in_channels_static=p["in_channels_static"],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=p["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=p["static_input_scalers_epsilon"],
        output_scalers=out_sig ** 0.5,
        n_lats_px=n_lats_px,
        n_lons_px=n_lons_px,
        patch_size_px=p["patch_size_px"],
        mask_unit_size_px=mask_unit_size_px,
        mask_ratio_inputs=0.0,  # no masking -- we want every token for embedding extraction
        mask_ratio_targets=0.0,  # required on the repo's main branch; >0 raises NotImplementedError (target masking unsupported)
        embed_dim=embed_dim,
        n_blocks_encoder=p["n_blocks_encoder"],
        n_blocks_decoder=p["n_blocks_decoder"],
        mlp_multiplier=p["mlp_multiplier"],
        n_heads=p["n_heads"],
        dropout=p["dropout"],
        drop_path=p["drop_path"],
        parameter_dropout=p["parameter_dropout"],
        residual="climate",  # mandatory -- pretrained static-embedding weights require it, see module docstring
        masking_mode="global",  # matches the repo's own current demo config; inconsequential when mask_ratio=0 either way
        positional_encoding="fourier",
        encoder_shifting=True,  # matches the repo's own current demo config for this exact checkpoint
        decoder_shifting=True,
        checkpoint_encoder=[],
        checkpoint_decoder=[],
    )

    # download_prithvi_weights.py mirrors the HF repo's own layout directly
    # under --weights-dir (the .pt is at the repo root, no "weights/"
    # subfolder -- that naming is the official notebook's own local
    # convention, not ours). Confirmed via sample_validation.py.
    weights_path = args.weights_dir / "prithvi.wxc.2300m.v1.pt"
    if not weights_path.exists():
        weights_path = args.weights_dir / "weights" / "prithvi.wxc.2300m.v1.pt"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    # The decoder shifter's local_mask/global_mask are buffers computed from
    # the grid shape, not learned weights. The checkpoint's copies are sized
    # for the global grid, so keep the ones this model built for the regional
    # grid and require every other key to load exactly.
    mask_keys = {k for k in state_dict if k.endswith(("shifter.local_mask", "shifter.global_mask"))}
    state_dict = {k: v for k, v in state_dict.items() if k not in mask_keys}
    result = model.load_state_dict(state_dict, strict=False)
    if set(result.missing_keys) - mask_keys or result.unexpected_keys:
        raise SystemExit(
            f"Unexpected weight mismatch. Missing: {sorted(set(result.missing_keys) - mask_keys)}; "
            f"unexpected: {result.unexpected_keys}"
        )
    print(f"Loaded weights from {weights_path} (using this grid's own {len(mask_keys)} shifter masks)")
    model = model.to(device)
    model.eval()

    # Keep the mask units in map order. With no masking, the model's own
    # mask generator still SHUFFLES the mask units into a random order for
    # every sample before the encoder (model.generate_mask -> _gen_mask_global).
    # The encoder output then comes back in that random order, and the
    # encoder's shifted-window attention, which assumes map order, mixes the
    # wrong neighbours. Runs before 2026-10-13 had this: pooled embeddings
    # were computed with scrambled geometry and the saved patch tokens sat in
    # the wrong quadrant three times out of four. The repo's masked_inference()
    # uses map order when no mask is given; do the same here.
    def ordered_mask(sizes):
        idx = model._global_idx.view(1, -1).expand(sizes[0], -1)
        return idx[:, :0], idx

    model.generate_mask = ordered_mask

    captured = {}

    def hook(_module, _inp, out):
        captured["x_encoded"] = out

    model.encoder.register_forward_hook(hook)

    # Optional taps inside the encoder. Only blocks 0, 4, 8, ... are offered:
    # those are local-attention blocks whose output is in the same token
    # layout as the final output (other blocks are transposed and/or have the
    # window shift applied).
    n_blocks_total = len(model.encoder.lgl_block.transformers)
    for name in args.layers:
        if name == "embed":
            model.encoder.register_forward_pre_hook(lambda _m, inp: captured.__setitem__("embed", inp[0]))
        elif name.isdigit() and int(name) % 4 == 0 and int(name) < n_blocks_total - 1:
            model.encoder.lgl_block.transformers[int(name)].register_forward_hook(
                lambda _m, _i, out, name=name: captured.__setitem__(name, out))
        else:
            raise SystemExit(f"--layers: '{name}' is not 'embed' or a multiple of 4 below {n_blocks_total - 1}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=functools.partial(preproc, padding=padding),
    )

    centers = region_centers(grid_lats.min(), grid_lons.min(), mask_unit_size_px, n_global_lat, n_global_lon)
    # One embedding per mask unit is only practical for a small crop (the globe has 216 units).
    save_units = n_global_mu <= 16

    # Patch-centre coordinates, and the box the pooled embedding is averaged over.
    patch_lat, patch_lon = p["patch_size_px"]
    lat_c = grid_lats.reshape(-1, patch_lat).mean(axis=1)
    lon_c = grid_lons.reshape(-1, patch_lon).mean(axis=1)

    def patches_in(box, what):
        lat_lo, lat_hi, lon_lo, lon_hi = box
        rows = np.where((lat_c >= lat_lo) & (lat_c < lat_hi))[0]
        cols = np.where((lon_c >= lon_lo) & (lon_c < lon_hi))[0]
        if len(rows) == 0 or len(cols) == 0:
            raise SystemExit(f"{what} {list(box)} contains no patches (grid spans lat {lat_c.min()}..{lat_c.max()}, lon {lon_c.min()}..{lon_c.max()}).")
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1

    q0, q1, d0, d1 = patches_in(args.pool_box, "--pool-box")
    print(f"Pooled embedding (global_emb_*): mean of {q1 - q0} x {d1 - d0} patches, lat centres {lat_c[q0]:.2f}..{lat_c[q1 - 1]:.2f}, "
          f"lon centres {lon_c[d0]:.3f}..{lon_c[d1 - 1]:.3f}")

    n_samples = len(dataset)

    # Un-pooled tokens for a sub-box. The pooled outputs below average over
    # the whole 30 x 40 degree crop, which washes out local weather; the
    # patch tokens (1 x 1.25 degrees each) keep it.
    tokens_out = None
    if args.out_tokens is not None:
        r0, r1, c0, c1 = patches_in(args.token_box, "--token-box")
        args.out_tokens.parent.mkdir(parents=True, exist_ok=True)
        tokens_out = np.lib.format.open_memmap(args.out_tokens, mode="w+", dtype=np.float16, shape=(n_samples, r1 - r0, c1 - c0, embed_dim))
        print(f"Saving patch tokens: {r1 - r0} x {c1 - c0} patches, lat centres {lat_c[r0]:.2f}..{lat_c[r1 - 1]:.2f}, "
              f"lon centres {lon_c[c0]:.3f}..{lon_c[c1 - 1]:.3f} ({tokens_out.nbytes / 1e9:.2f} GB)")
        stem = args.out_tokens.with_suffix("")
        layer_out = {name: np.lib.format.open_memmap(f"{stem}_layer{name}.npy", mode="w+", dtype=np.float16, shape=tokens_out.shape)
                     for name in args.layers}
        # The same box in pixels, for the model's inputs and its reconstruction.
        px = (slice(None), slice(None), slice(r0 * patch_lat, r1 * patch_lat), slice(c0 * patch_lon, c1 * patch_lon))
        n_px_lat, n_px_lon = (r1 - r0) * patch_lat, (c1 - c0) * patch_lon
        n_sfc = len(SURFACE_VARS)
        inputs_out = np.lib.format.open_memmap(f"{stem}_inputs.npy", mode="w+", dtype=np.float16, shape=(n_samples, p["in_channels"], n_px_lat, n_px_lon))
        recon_out = np.lib.format.open_memmap(f"{stem}_recon_sfc.npy", mode="w+", dtype=np.float16, shape=(n_samples, n_sfc, n_px_lat, n_px_lon))
        climate_out = np.lib.format.open_memmap(f"{stem}_climate_sfc.npy", mode="w+", dtype=np.float16, shape=(n_samples, n_sfc, n_px_lat, n_px_lon))
        wide = None
        if args.wide_box is not None:
            w_lat = np.where((grid_lats >= args.wide_box[0]) & (grid_lats <= args.wide_box[1]))[0]
            w_lon = np.where((grid_lons >= args.wide_box[2]) & (grid_lons <= args.wide_box[3]))[0]
            wide_ch = [SURFACE_VARS.index(v) for v in ("T2M", "QV2M", "U10M", "V10M", "SWGNT")]
            wide_px = (slice(None), wide_ch, slice(int(w_lat[0]), int(w_lat[-1]) + 1), slice(int(w_lon[0]), int(w_lon[-1]) + 1))
            wshape = (n_samples, len(wide_ch), len(w_lat), len(w_lon))
            wide = {"recon": np.lib.format.open_memmap(f"{stem}_wide_recon5.npy", mode="w+", dtype=np.float16, shape=wshape),
                    "inputs": np.lib.format.open_memmap(f"{stem}_wide_inputs5.npy", mode="w+", dtype=np.float16, shape=wshape)}
            print(f"Wide box: {wshape[2]} x {wshape[3]} pixels, lat {grid_lats[w_lat[0]]}..{grid_lats[w_lat[-1]]}, "
                  f"lon {grid_lons[w_lon[0]]}..{grid_lons[w_lon[-1]]} ({2 * np.prod(wshape) * 2 / 1e9:.2f} GB for both files)")
        pooled = wide_tok = None
        if args.pooled:
            ext_patches, ext_px = {}, {}
            for name, box in extents.items():
                ext_patches[name] = patches_in(box, f"pool extent {name}")
                la = np.where((grid_lats >= box[0]) & (grid_lats <= box[1]))[0]
                lo = np.where((grid_lons >= box[2]) & (grid_lons <= box[3]))[0]
                ext_px[name] = (slice(int(la[0]), int(la[-1]) + 1), slice(int(lo[0]), int(lo[-1]) + 1))
                e = ext_patches[name]
                print(f"  pool extent {name:12s}: {e[1] - e[0]} x {e[3] - e[2]} patches, {len(la)} x {len(lo)} pixels")
            n_ext, n_ch = len(extents), p["in_channels"]
            pooled = {"tokens": np.lib.format.open_memmap(f"{stem}_pooled_tokens.npy", mode="w+", dtype=np.float16, shape=(n_samples, n_ext, embed_dim)),
                      "inputs": np.lib.format.open_memmap(f"{stem}_pooled_inputs.npy", mode="w+", dtype=np.float32, shape=(n_samples, n_ext, n_ch)),
                      "recon": np.lib.format.open_memmap(f"{stem}_pooled_recon.npy", mode="w+", dtype=np.float32, shape=(n_samples, n_ext, n_ch))}
        if args.save_wide_patch_tokens:
            wr0, wr1, wc0, wc1 = patches_in(POOL_EXTENTS["wide"], "wide patch tokens")
            wide_tok = np.lib.format.open_memmap(f"{stem}_wide_tokens.npy", mode="w+", dtype=np.float16, shape=(n_samples, wr1 - wr0, wc1 - wc0, embed_dim))
            print(f"Saving wide patch tokens: {wr1 - wr0} x {wc1 - wc0} patches ({wide_tok.nbytes / 1e9:.2f} GB)")
        norm_mu = model.input_scalers_mu.reshape(1, -1, 1, 1)
        norm_sd = model.input_scalers_sigma.reshape(1, -1, 1, 1) + model.input_scalers_epsilon
        print(f"Also saving: normalised inputs {inputs_out.shape}, decoder output and climatology for the {n_sfc} surface variables, "
              f"encoder layers {args.layers or 'none'}")
    local_lat, local_lon = model.local_shape_mu

    global_out = np.empty((n_samples, embed_dim), dtype=np.float32)
    region_out = np.empty((n_samples, n_global_mu, embed_dim), dtype=np.float32) if save_units else None

    idx = 0
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # The loader and model have no missing-value handling: one NaN in
            # the inputs turns every embedding NaN. Stop at the first one.
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all():
                    n_bad = int((~torch.isfinite(v)).sum())
                    raise SystemExit(
                        f"Non-finite values in model input '{k}' ({n_bad}/{v.numel()}) at "
                        f"timestamps starting {timestamps[idx]}. Check the daily MERRA-2 files and climatology."
                    )

            if idx == 0:
                check_normalised_inputs(batch["x"], model, n_levels=len(LEVELS))

            model_out = model(batch)  # [B, channels, lat, lon], physical units
            x_encoded = captured["x_encoded"]  # [B, n_global_mu, n_local_mu, embed_dim]
            if idx == 0:
                print(f"First-batch embedding magnitude: mean |value| {float(x_encoded.abs().mean()):.3g}, max {float(x_encoded.abs().max()):.3g}")
            if not torch.isfinite(x_encoded).all():
                raise SystemExit(f"Model produced non-finite embeddings at timestamps starting {timestamps[idx]} despite finite inputs.")

            b = x_encoded.shape[0]
            global_out[idx:idx + b] = patch_grid(x_encoded, n_global_lat, n_global_lon, local_lat, local_lon)[:, q0:q1, d0:d1].mean(dim=(1, 2)).cpu().numpy()
            if save_units:
                region_out[idx:idx + b] = x_encoded.mean(dim=2).cpu().numpy()
            if tokens_out is not None:
                def boxed(tok):
                    return patch_grid(tok, n_global_lat, n_global_lon, local_lat, local_lon)[:, r0:r1, c0:c1].to(torch.float16).cpu().numpy()

                def f16(z):  # normalised fields; clamp so float16 cannot overflow on heavy-tailed variables
                    return z.clamp(-6e4, 6e4).to(torch.float16).cpu().numpy()

                tokens_out[idx:idx + b] = boxed(x_encoded)
                for name, arr in layer_out.items():
                    arr[idx:idx + b] = boxed(captured[name])
                x_now = (batch["x"][:, -1] - norm_mu) / norm_sd          # inputs at the embedding's own time
                recon = (model_out - norm_mu) / norm_sd
                inputs_out[idx:idx + b] = f16(x_now[px])
                recon_out[idx:idx + b] = f16(recon[px][:, :n_sfc])
                if wide is not None:
                    wide["recon"][idx:idx + b] = f16(recon[wide_px])
                    wide["inputs"][idx:idx + b] = f16(x_now[wide_px])
                climate_out[idx:idx + b] = f16(((batch["climate"] - norm_mu) / norm_sd)[px][:, :n_sfc])
                if pooled is not None or wide_tok is not None:
                    grid_tok = patch_grid(x_encoded, n_global_lat, n_global_lon, local_lat, local_lon)   # [B, rows, cols, C]
                if pooled is not None:
                    for e, name in enumerate(extents):
                        r0e, r1e, c0e, c1e = ext_patches[name]
                        pooled["tokens"][idx:idx + b, e] = grid_tok[:, r0e:r1e, c0e:c1e].mean(dim=(1, 2)).to(torch.float16).cpu().numpy()
                        pla, plo = ext_px[name]
                        pooled["inputs"][idx:idx + b, e] = x_now[:, :, pla, plo].mean(dim=(2, 3)).float().cpu().numpy()
                        pooled["recon"][idx:idx + b, e] = recon[:, :, pla, plo].mean(dim=(2, 3)).float().cpu().numpy()
                if wide_tok is not None:
                    wide_tok[idx:idx + b] = grid_tok[:, wr0:wr1, wc0:wc1].to(torch.float16).cpu().numpy()
                if idx == 0:
                    target = (batch["y"] - norm_mu) / norm_sd
                    i_t2m = SURFACE_VARS.index("T2M")
                    print(f"First batch, T2M in the box (standard deviations): decoder output vs target, mean |difference| "
                          f"{float((recon - target)[px][:, i_t2m].abs().mean()):.3f}; input at t vs target "
                          f"{float((x_now - target)[px][:, i_t2m].abs().mean()):.3f}; climatology vs target "
                          f"{float(((batch['climate'] - norm_mu) / norm_sd - target)[px][:, i_t2m].abs().mean()):.3f}")
            idx += b

            if idx % 100 < b:
                print(f"...{idx}/{n_samples} timestamps embedded")

    columns = ["timestamp"]
    columns += [f"global_emb_{d}" for d in range(embed_dim)]
    if save_units:
        columns += [f"region{r}_emb_{d}" for r in range(n_global_mu) for d in range(embed_dim)]
        data = np.concatenate([global_out, region_out.reshape(n_samples, -1)], axis=1)
    else:
        data = global_out
    df = pd.DataFrame(data, columns=columns[1:])
    df.insert(0, "timestamp", pd.to_datetime(timestamps))

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out_parquet}")

    if tokens_out is not None:
        for arr in [tokens_out, inputs_out, recon_out, climate_out, *layer_out.values(), *(wide.values() if wide else []),
                    *(pooled.values() if pooled else []), *([wide_tok] if wide_tok is not None else [])]:
            arr.flush()
        meta = {
            "layout": "[time, lat_patch, lon_patch, embed_dim], float16; encoder output tokens, un-pooled, mask units in map order",
            "block_order": "map",
            "layers": {name: f"{stem.name}_layer{name}.npy" for name in args.layers},
            "fields": {
                "inputs": {"file": f"{stem.name}_inputs.npy", "layout": "[time, channel, lat_pixel, lon_pixel], normalised with the model's input mean and sigma, at the embedding's own time",
                           "channels": list(SURFACE_VARS) + [f"{v}_{int(l)}" for v in VERTICAL_VARS for l in LEVELS]},
                "recon_sfc": {"file": f"{stem.name}_recon_sfc.npy", "layout": "decoder output, surface channels only, same normalisation", "channels": list(SURFACE_VARS)},
                "climate_sfc": {"file": f"{stem.name}_climate_sfc.npy", "layout": "climatology given to the model for the target time, surface channels, same normalisation", "channels": list(SURFACE_VARS)},
            },
            "pixel_lats": grid_lats[r0 * patch_lat:r1 * patch_lat].tolist(),
            "pixel_lons": grid_lons[c0 * patch_lon:c1 * patch_lon].tolist(),
            "model_grid": f"{n_lats_px} x {n_lons_px}",
            "pool_box": list(args.pool_box),
            "input_time_hours": args.input_time,
            "lead_time_hours": args.lead_time,
            "timestamps": [pd.Timestamp(ts).isoformat() for ts in timestamps],
            "patch_lat_centers": lat_c[r0:r1].tolist(),
            "patch_lon_centers": lon_c[c0:c1].tolist(),
            "token_box": list(args.token_box),
            **({"wide": {"box": list(args.wide_box), "channels": ["T2M", "QV2M", "U10M", "V10M", "SWGNT"],
                         "recon_file": f"{stem.name}_wide_recon5.npy", "inputs_file": f"{stem.name}_wide_inputs5.npy",
                         "pixel_lats": grid_lats[wide_px[2]].tolist(), "pixel_lons": grid_lons[wide_px[3]].tolist(),
                         "layout": "[time, channel, lat_pixel, lon_pixel], normalised; recon = decoder output for the target time, "
                                   "inputs = true values at the issue time"}} if wide else {}),
            **({"pooled": {"extents": {k: list(v) for k, v in extents.items()}, "pool_set": args.pool_set,
                           "tokens_file": f"{stem.name}_pooled_tokens.npy", "inputs_file": f"{stem.name}_pooled_inputs.npy",
                           "recon_file": f"{stem.name}_pooled_recon.npy",
                           "channels": list(SURFACE_VARS) + [f"{v}_{int(l)}" for v in VERTICAL_VARS for l in LEVELS],
                           "layout": "tokens [time, extent, embed_dim] float16, final encoder layer averaged over the extent's patches "
                                     "(centres inside the box); inputs/recon [time, extent, channel] float32, normalised with the "
                                     "model's input mean and sigma, averaged over the extent's pixels; inputs = issue time, "
                                     "recon = decoder output for the target time (issue + lead)"}} if pooled else {}),
            **({"wide_tokens": {"file": f"{stem.name}_wide_tokens.npy", "box": list(POOL_EXTENTS["wide"]),
                                "patch_lat_centers": lat_c[wr0:wr1].tolist(), "patch_lon_centers": lon_c[wc0:wc1].tolist()}}
               if wide_tok is not None else {}),
        }
        args.out_tokens.with_suffix(".json").write_text(json.dumps(meta))
        print(f"Wrote patch tokens {tokens_out.shape} to {args.out_tokens}")

    meta_path = args.out_parquet.with_suffix(".regions.csv")
    pd.DataFrame(centers, columns=["lat_center", "lon_center"]).reset_index(names="region_index").to_csv(meta_path, index=False)
    print(f"Wrote region geographic centers to {meta_path}")


if __name__ == "__main__":
    main()
