"""Does the pretrained Prithvi WxC model actually work the way we run it?

Runs the model on the official GLOBAL sample data that ships with the
weights (2020), and scores its output against the target field, relative to
the climatology the model is handed:

    skill = 1 - MSE(model output - target) / MSE(climatology - target)

1 is perfect, 0 is no better than climatology, negative is worse. With lead
time 0 the target equals the model's own input, so a working model should
score close to 1.

The same samples are run on crops of increasing size around New England,
to find how much surrounding area the model needs to work properly:

    2 x 2 mask units   30-60N, 90-50W (the crop all runs used up to 2026-10-13)
    4 x 4              15-75N, 110-30W
    4 x 6              15-75N, 130-10W
    6 x 8              0-90N, 150W-10E
    12 x 18            the whole globe (what the model was trained on)

Every row is scored inside the same two areas so they are comparable: the
original 2 x 2 box, and a small New England box at its centre (where the
tokens we use come from; crop edges are far from it in the larger crops).

Findings so far (2026-10-13): on the global grid skill is 0.99 at lead 0 and
0.96 at lead 6 h; on the 2 x 2 crop 0.89 and 0.49. The order of the mask
units (shuffled as in the repo's default forward pass, or map order) makes
no difference to skill; switching off the shifted-window attention hurts.

Uses only the repo's own loader and model code, on data already downloaded
by cluster/sbatch/service/01_setup_prithvi.sbatch.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from PrithviWxC.dataloaders.merra2 import (
    Merra2Dataset,
    input_scalers,
    output_scalers,
    preproc,
    static_input_scalers,
)
from PrithviWxC.model import PrithviWxC, _Shift

SURFACE_VARS = [
    "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP", "PS",
    "QV2M", "SLP", "SWGNT", "SWTNT", "T2M", "TQI", "TQL", "TQV", "TS",
    "U10M", "V10M", "Z0M",
]
STATIC_SURFACE_VARS = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
VERTICAL_VARS = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
LEVELS = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0]
GLOBAL_PADDING = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}  # matches the official demo: trims the duplicate 361st lat row


# Crops are given as (name, first lat row, lat rows, first lon column, lon columns)
# on the 360 x 576 global grid (lat = -90 + 0.5 * row, lon = -180 + 0.625 * column).
# Rows and columns must be multiples of the mask unit (30 x 32 pixels).
CROPS = [
    ("2 x 2 units (30-60N, 90-50W)", 240, 60, 144, 64),
    ("4 x 4 units (15-75N, 110-30W)", 210, 120, 112, 128),
    ("4 x 6 units (15-75N, 130-10W)", 210, 120, 80, 192),
    ("6 x 8 units (0-90N, 150W-10E)", 180, 180, 48, 256),
    ("12 x 18 units (global)", 0, 360, 0, 576),
]
SCORE_AREAS = {"2 x 2 box": (240, 60, 144, 64), "New England": (260, 16, 168, 14)}


def build_model(p, scalers, n_lats_px, n_lons_px):
    in_mu, in_sig, out_sig, static_mu, static_sig = scalers
    return PrithviWxC(
        in_channels=p["in_channels"], input_size_time=p["input_size_time"], in_channels_static=p["in_channels_static"],
        input_scalers_mu=in_mu, input_scalers_sigma=in_sig, input_scalers_epsilon=p["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu, static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=p["static_input_scalers_epsilon"],
        output_scalers=out_sig ** 0.5,
        n_lats_px=n_lats_px, n_lons_px=n_lons_px,
        patch_size_px=p["patch_size_px"], mask_unit_size_px=p["mask_unit_size_px"],
        mask_ratio_inputs=0.0, mask_ratio_targets=0.0,
        embed_dim=p["embed_dim"], n_blocks_encoder=p["n_blocks_encoder"], n_blocks_decoder=p["n_blocks_decoder"],
        mlp_multiplier=p["mlp_multiplier"], n_heads=p["n_heads"],
        dropout=p["dropout"], drop_path=p["drop_path"], parameter_dropout=p["parameter_dropout"],
        residual="climate", masking_mode="global", positional_encoding="fourier",
        encoder_shifting=True, decoder_shifting=True, checkpoint_encoder=[], checkpoint_decoder=[],
    )


def load_weights(model, state_dict, regional: bool) -> None:
    if regional:
        # Shifter masks are buffers computed from the grid shape; keep the regional model's own.
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(("shifter.local_mask", "shifter.global_mask"))}
        result = model.load_state_dict(state_dict, strict=False)
        extra = [k for k in result.missing_keys if not k.endswith(("local_mask", "global_mask"))]
        if extra or result.unexpected_keys:
            raise SystemExit(f"Weight mismatch: missing {extra}, unexpected {result.unexpected_keys}")
    else:
        model.load_state_dict(state_dict, strict=True)


def configure(model, ordered: bool, shift: bool) -> None:
    """Switch block order and window shifting on an existing model."""
    if ordered:
        def ordered_mask(sizes):
            idx = model._global_idx.view(1, -1).expand(sizes[0], -1)
            return idx[:, :0], idx
        model.generate_mask = ordered_mask
    else:
        model.generate_mask = model._gen_mask_global
    if not hasattr(model, "_orig_shifters"):
        model._orig_shifters = (model.encoder.lgl_block.shifter, model.decoder.lgl_block.shifter)
    enc, dec = model._orig_shifters if shift else (_Shift(), _Shift())
    model.encoder.lgl_block.shifter = enc
    model.decoder.lgl_block.shifter = dec


class Score:
    """Running per-channel squared error of the model output and of climatology."""

    def __init__(self, n_channels: int):
        self.se_model = torch.zeros(n_channels, dtype=torch.float64)
        self.se_clim = torch.zeros(n_channels, dtype=torch.float64)
        self.abs_t2m = 0.0
        self.n_px = 0

    def add(self, out, y, clim, t2m_sigma: float) -> None:
        self.se_model += ((out - y) ** 2).sum(dim=(0, 2, 3)).double().cpu()
        self.se_clim += ((clim - y) ** 2).sum(dim=(0, 2, 3)).double().cpu()
        i = SURFACE_VARS.index("T2M")
        self.abs_t2m += float((out[:, i] - y[:, i]).abs().sum()) / t2m_sigma
        self.n_px += out.shape[0] * out.shape[2] * out.shape[3]

    def summary(self) -> dict:
        skill = 1 - self.se_model / self.se_clim.clamp(min=1e-30)
        n_sfc = len(SURFACE_VARS)
        i = SURFACE_VARS.index("T2M")
        return {
            "surface (median of 20)": float(skill[:n_sfc].median()),
            "vertical (median of 140)": float(skill[n_sfc:].median()),
            "T2M skill": float(skill[i]),
            "T2M mean |err| (sd)": self.abs_t2m / self.n_px,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", type=Path, required=True, help="Downloaded Prithvi-WxC/prithvi.wxc.2300m.v1 snapshot")
    parser.add_argument("--start", default="2020-01-01T00:00:00")
    parser.add_argument("--end", default="2020-01-03T23:59:59")
    parser.add_argument("--input-time", type=int, default=-6)
    parser.add_argument("--lead-times", type=int, nargs="+", default=[0, 6])
    parser.add_argument("--n-samples", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    config = yaml.safe_load((args.weights_dir / "config.yaml").read_text())
    p = config["params"]
    clim_dir = args.weights_dir / "climatology"
    in_mu, in_sig = input_scalers(SURFACE_VARS, VERTICAL_VARS, LEVELS, clim_dir / "musigma_surface.nc", clim_dir / "musigma_vertical.nc")
    out_sig = output_scalers(SURFACE_VARS, VERTICAL_VARS, LEVELS, clim_dir / "anomaly_variance_surface.nc", clim_dir / "anomaly_variance_vertical.nc")
    static_mu, static_sig = static_input_scalers(clim_dir / "musigma_surface.nc", STATIC_SURFACE_VARS)
    scalers = (in_mu, in_sig, out_sig, static_mu, static_sig)
    t2m_sigma = float(in_sig[SURFACE_VARS.index("T2M")])

    weights_path = args.weights_dir / "prithvi.wxc.2300m.v1.pt"
    if not weights_path.exists():
        weights_path = args.weights_dir / "weights" / "prithvi.wxc.2300m.v1.pt"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    # Load the samples once per lead time (CPU), reuse them for every configuration.
    batches = {}
    for lead in args.lead_times:
        dataset = Merra2Dataset(
            time_range=(args.start, args.end), lead_times=[lead], input_times=[args.input_time],
            data_path_surface=args.weights_dir / "merra-2", data_path_vertical=args.weights_dir / "merra-2",
            climatology_path_surface=clim_dir, climatology_path_vertical=clim_dir,
            surface_vars=SURFACE_VARS, static_surface_vars=STATIC_SURFACE_VARS, vertical_vars=VERTICAL_VARS,
            levels=LEVELS, positional_encoding="fourier",
        )
        if len(dataset) == 0:
            raise SystemExit("No valid samples in the official sample data for this time range.")
        step = max(1, len(dataset) // args.n_samples)
        batches[lead] = [preproc([dataset[i]], GLOBAL_PADDING) for i in range(0, len(dataset), step)][: args.n_samples]
        lats = np.asarray(dataset.lats)[: batches[lead][0]["x"].shape[-2]]
        lons = np.asarray(dataset.lons)
        print(f"Lead {lead} h: {len(batches[lead])} samples of {len(dataset)}, grid {batches[lead][0]['x'].shape[-2]} x {batches[lead][0]['x'].shape[-1]}", flush=True)

    print("Score areas: " + "; ".join(f"{k}: lat {lats[r0]}..{lats[r0 + nr - 1]}, lon {lons[c0]}..{lons[c0 + nc - 1]}" for k, (r0, nr, c0, nc) in SCORE_AREAS.items()), flush=True)

    results = []
    for name, r0, nr, c0, nc in CROPS:
        regional = (nr, nc) != (len(lats), len(lons))
        model = build_model(p, scalers, nr, nc)
        load_weights(model, state_dict, regional)
        model = model.to(device).eval()
        configure(model, ordered=True, shift=True)
        for lead in args.lead_times:
            scores = {k: Score(p["in_channels"]) for k in SCORE_AREAS}
            for cpu_batch in batches[lead]:
                b = {k: (v[..., r0:r0 + nr, c0:c0 + nc].to(device) if isinstance(v, torch.Tensor) and v.ndim >= 3 else (v.to(device) if isinstance(v, torch.Tensor) else v))
                     for k, v in cpu_batch.items()}
                with torch.no_grad():
                    out = model(b)
                if not torch.isfinite(out).all():
                    raise SystemExit("Model output is not finite.")
                for k, (a0, an, b0, bn) in SCORE_AREAS.items():
                    sl = (Ellipsis, slice(a0 - r0, a0 - r0 + an), slice(b0 - c0, b0 - c0 + bn))
                    scores[k].add(out[sl], b["y"][sl], b["climate"][sl], t2m_sigma)
            results.append((name, lead, {k: s.summary() for k, s in scores.items()}))
            s = results[-1][2]["New England"]
            print(f"  [{name}] lead {lead} h, New England: surface {s['surface (median of 20)']:.3f}, vertical {s['vertical (median of 140)']:.3f}, T2M {s['T2M skill']:.3f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for area in SCORE_AREAS:
        print(f"\nSkill relative to climatology (1 = perfect, 0 = no better than climatology), scored in: {area}")
        print(f"{'crop given to the model':34s} {'lead':>5s} {'surface':>8s} {'vertical':>9s} {'T2M':>7s} {'T2M |err| sd':>13s}")
        for name, lead, summ in results:
            s = summ[area]
            print(f"{name:34s} {lead:4d}h {s['surface (median of 20)']:8.3f} {s['vertical (median of 140)']:9.3f} {s['T2M skill']:7.3f} {s['T2M mean |err| (sd)']:13.4f}")


if __name__ == "__main__":
    main()
