"""Does the pretrained Prithvi WxC model actually work the way we run it?

Runs the model on the official GLOBAL sample data that ships with the
weights (2020), and scores its output against the target field, relative to
the climatology the model is handed:

    skill = 1 - MSE(model output - target) / MSE(climatology - target)

1 is perfect, 0 is no better than climatology, negative is worse. With lead
time 0 the target equals the model's own input, so a working model should
score close to 1.

The same samples are run several ways, to separate out which of our choices
(if any) breaks the model:

    global, shuffled blocks   the repo's standard forward pass (what its
                              inference notebook does, and what our runs did
                              before 2026-10-13)
    global, map order         mask units kept in map order
    global, map order,        the same with the encoder's and decoder's
      no window shift         shifted-window attention switched off
    regional crop, map order  OUR SETUP: the same samples cropped to the
                              30-59.5N, 90-50.6W box, 2 x 2 mask units
    regional crop, shuffled   our setup before the block-order fix

Scores are reported over the whole grid where that applies, and over our
box in every case so the rows are comparable.

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


BOX = (30.0, 59.5, -90.0, -50.625)  # lat_min, lat_max, lon_min, lon_max of our regional crop


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
            "worst channel": float(skill.min()),
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

    rows = np.where((lats >= BOX[0]) & (lats <= BOX[1]))[0]
    cols = np.where((lons >= BOX[2]) & (lons <= BOX[3]))[0]
    r, c = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
    print(f"Our box inside the global grid: {len(rows)} x {len(cols)} pixels, lat {lats[rows[0]]}..{lats[rows[-1]]}, lon {lons[cols[0]]}..{lons[cols[-1]]}", flush=True)
    if (len(rows), len(cols)) != (60, 64):
        raise SystemExit("Box does not come out as 60 x 64 pixels; check the sample grid.")

    def crop(batch):
        return {k: (v[..., r, c] if isinstance(v, torch.Tensor) and v.ndim >= 3 else v) for k, v in batch.items()}

    def run(model, lead, regional: bool) -> tuple[Score, Score]:
        whole, box = Score(p["in_channels"]), Score(p["in_channels"])
        for cpu_batch in batches[lead]:
            b = crop(cpu_batch) if regional else cpu_batch
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.no_grad():
                out = model(b)
            if not torch.isfinite(out).all():
                raise SystemExit("Model output is not finite.")
            if regional:
                box.add(out, b["y"], b["climate"], t2m_sigma)
            else:
                whole.add(out, b["y"], b["climate"], t2m_sigma)
                box.add(out[..., r, c], b["y"][..., r, c], b["climate"][..., r, c], t2m_sigma)
        return whole, box

    results = []
    for regional, n_lat, n_lon in ((False, len(lats), len(lons)), (True, 60, 64)):
        model = build_model(p, scalers, n_lat, n_lon)
        load_weights(model, state_dict, regional)
        model = model.to(device).eval()
        configs = ([("global, shuffled blocks (repo default)", False, True), ("global, map order", True, True), ("global, map order, no window shift", True, False)]
                   if not regional else
                   [("REGIONAL crop, map order (our setup)", True, True), ("regional crop, shuffled (our old setup)", False, True), ("regional crop, map order, no window shift", True, False)])
        for name, ordered, shift in configs:
            configure(model, ordered, shift)
            for lead in args.lead_times:
                torch.manual_seed(0)
                whole, box = run(model, lead, regional)
                results.append((name, lead, "whole grid" if not regional else None, whole, box))
                s = box.summary()
                print(f"  [{name}] lead {lead} h, in our box: surface {s['surface (median of 20)']:.3f}, vertical {s['vertical (median of 140)']:.3f}, T2M {s['T2M skill']:.3f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nSkill relative to climatology (1 = perfect, 0 = no better than climatology), scored inside our box:")
    print(f"{'configuration':44s} {'lead':>5s} {'surface':>8s} {'vertical':>9s} {'T2M':>7s} {'T2M |err| sd':>13s} {'worst ch.':>10s}")
    for name, lead, _, whole, box in results:
        s = box.summary()
        print(f"{name:44s} {lead:4d}h {s['surface (median of 20)']:8.3f} {s['vertical (median of 140)']:9.3f} {s['T2M skill']:7.3f} {s['T2M mean |err| (sd)']:13.4f} {s['worst channel']:10.2f}")
    print("\nSame, scored over the whole globe (global runs only):")
    for name, lead, tag, whole, box in results:
        if tag:
            s = whole.summary()
            print(f"{name:44s} {lead:4d}h {s['surface (median of 20)']:8.3f} {s['vertical (median of 140)']:9.3f} {s['T2M skill']:7.3f} {s['T2M mean |err| (sd)']:13.4f} {s['worst channel']:10.2f}")


if __name__ == "__main__":
    main()
