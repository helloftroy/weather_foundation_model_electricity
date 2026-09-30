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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    n_lats_px = len(dataset.lats)
    n_lons_px = len(dataset.lons)
    print(f"Regional grid: {n_lats_px} x {n_lons_px} pixels")

    config = yaml.safe_load((args.weights_dir / "config.yaml").read_text())
    p = config["params"]
    mask_unit_size_px = p["mask_unit_size_px"]
    if n_lats_px % mask_unit_size_px[0] != 0 or n_lons_px % mask_unit_size_px[1] != 0:
        raise SystemExit(
            f"Regional grid {n_lats_px}x{n_lons_px} is not divisible by "
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
        embed_dim=embed_dim,
        n_blocks_encoder=p["n_blocks_encoder"],
        n_blocks_decoder=p["n_blocks_decoder"],
        mlp_multiplier=p["mlp_multiplier"],
        n_heads=p["n_heads"],
        dropout=p["dropout"],
        drop_path=p["drop_path"],
        parameter_dropout=p["parameter_dropout"],
        residual="climate",  # mandatory -- pretrained static-embedding weights require it, see module docstring
        masking_mode="local",
        positional_encoding="fourier",
        decoder_shifting=False,
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
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    captured = {}

    def hook(_module, _inp, out):
        captured["x_encoded"] = out

    model.encoder.register_forward_hook(hook)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: preproc(batch, NO_PADDING),
    )

    centers = region_centers(dataset.lats.min(), dataset.lons.min(), mask_unit_size_px, n_global_lat, n_global_lon)

    n_samples = len(dataset)
    global_out = np.empty((n_samples, embed_dim), dtype=np.float32)
    region_out = np.empty((n_samples, n_global_mu, embed_dim), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            model(batch)  # decoder output discarded; we only need the hook capture
            x_encoded = captured["x_encoded"]  # [B, n_global_mu, n_local_mu, embed_dim]

            b = x_encoded.shape[0]
            global_out[idx:idx + b] = x_encoded.mean(dim=(1, 2)).cpu().numpy()
            region_out[idx:idx + b] = x_encoded.mean(dim=2).cpu().numpy()
            idx += b

            if idx % 100 < b:
                print(f"...{idx}/{n_samples} timestamps embedded")

    columns = ["timestamp"]
    columns += [f"global_emb_{d}" for d in range(embed_dim)]
    columns += [f"region{r}_emb_{d}" for r in range(n_global_mu) for d in range(embed_dim)]

    data = np.concatenate([global_out, region_out.reshape(n_samples, -1)], axis=1)
    df = pd.DataFrame(data, columns=columns[1:])
    df.insert(0, "timestamp", pd.to_datetime(timestamps))

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out_parquet}")

    meta_path = args.out_parquet.with_suffix(".regions.csv")
    pd.DataFrame(centers, columns=["lat_center", "lon_center"]).reset_index(names="region_index").to_csv(meta_path, index=False)
    print(f"Wrote region geographic centers to {meta_path}")


if __name__ == "__main__":
    main()
