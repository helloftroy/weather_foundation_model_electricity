"""Official-pipeline sample validation, using data we already have.

Replaces running examples/PrithviWxC_inference.ipynb directly: that
notebook's own hf_hub_download/snapshot_download cells point at
ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M, whose merra-2/ and climatology/
directories are empty upstream (confirmed via direct HTTP HEAD checks,
2026-09-30) -- likely deprecated in favor of Prithvi-WxC/prithvi.wxc.2300m.v1,
which IS fully populated (confirmed: surface, vertical, and all climatology/
scaler files present for the full 2020 sample year) and is the exact repo
Phase 1 already downloads in full via cluster/sbatch/service/01_setup_prithvi.sbatch.

So instead of re-downloading (and hitting the same dead directories the
notebook does), this points the OFFICIAL Merra2Dataset/PrithviWxC/preproc
code directly at our already-downloaded weights-dir's merra-2/ and
climatology/ subdirectories. Same official preprocessing and model code as
the notebook -- just correctly located data, and it also directly exercises
the encoder-hook embedding extraction from src/prithvi/extract_embeddings.py
against real weights as a bonus sanity check.

Uses the GLOBAL grid (the sample data is global, not our regional crop) with
a tiny 2-day time range, matching the official demo's own scope.
"""
import argparse
from pathlib import Path

import torch
import yaml

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
GLOBAL_PADDING = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}  # matches the official demo: trims the duplicate 361st lat row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True, help="Downloaded Prithvi-WxC/prithvi.wxc.2300m.v1 snapshot")
    parser.add_argument("--start", default="2020-01-01T00:00:00")
    parser.add_argument("--end", default="2020-01-02T05:59:59")
    parser.add_argument("--input-time", type=int, default=-6)
    parser.add_argument("--lead-time", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.0, help="0.0 matches our Phase 2 embedding-extraction use case")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    surf_dir = args.weights_dir / "merra-2"
    vert_dir = args.weights_dir / "merra-2"
    clim_surf_dir = args.weights_dir / "climatology"
    clim_vert_dir = args.weights_dir / "climatology"

    dataset = Merra2Dataset(
        time_range=(args.start, args.end),
        lead_times=[args.lead_time],
        input_times=[args.input_time],
        data_path_surface=surf_dir,
        data_path_vertical=vert_dir,
        climatology_path_surface=clim_surf_dir,
        climatology_path_vertical=clim_vert_dir,
        surface_vars=SURFACE_VARS,
        static_surface_vars=STATIC_SURFACE_VARS,
        vertical_vars=VERTICAL_VARS,
        levels=LEVELS,
        positional_encoding="fourier",
    )
    assert len(dataset) > 0, "No valid samples -- check the sample data actually downloaded (see script docstring)."
    print(f"Dataset: {len(dataset)} samples")

    # n_lats_px/n_lons_px must reflect the grid AFTER padding (preproc()
    # applies GLOBAL_PADDING, e.g. [0, -1] on lat trims the raw file's 361st
    # duplicate pole row down to 360) -- NOT the raw file's own coordinate
    # count, or the model constructor and the actual padded tensor shape
    # mismatch.
    n_lats_px = len(dataset.lats) + GLOBAL_PADDING["lat"][0] + GLOBAL_PADDING["lat"][1]
    n_lons_px = len(dataset.lons) + GLOBAL_PADDING["lon"][0] + GLOBAL_PADDING["lon"][1]
    print(f"Raw file grid: {len(dataset.lats)} x {len(dataset.lons)}")
    print(f"Post-padding grid (used for model construction): {n_lats_px} x {n_lons_px} (should be 360 x 576)")

    config = yaml.safe_load((args.weights_dir / "config.yaml").read_text())
    p = config["params"]

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
        mask_unit_size_px=p["mask_unit_size_px"],
        mask_ratio_inputs=args.mask_ratio,
        embed_dim=p["embed_dim"],
        n_blocks_encoder=p["n_blocks_encoder"],
        n_blocks_decoder=p["n_blocks_decoder"],
        mlp_multiplier=p["mlp_multiplier"],
        n_heads=p["n_heads"],
        dropout=p["dropout"],
        drop_path=p["drop_path"],
        parameter_dropout=p["parameter_dropout"],
        residual="climate",
        masking_mode="local",
        positional_encoding="fourier",
        decoder_shifting=False,
        checkpoint_encoder=[],
        checkpoint_decoder=[],
    )

    # src/prithvi/download_prithvi_weights.py mirrors the HF repo's own
    # layout directly under --weights-dir (the .pt is at the repo root,
    # since Prithvi-WxC/prithvi.wxc.2300m.v1 has no "weights/" subfolder --
    # that naming is the official notebook's own local convention, not ours).
    weights_path = args.weights_dir / "prithvi.wxc.2300m.v1.pt"
    if not weights_path.exists():
        weights_path = args.weights_dir / "weights" / "prithvi.wxc.2300m.v1.pt"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    captured = {}
    model.encoder.register_forward_hook(lambda _m, _i, out: captured.__setitem__("x_encoded", out))

    data = next(iter(dataset))
    batch = preproc([data], GLOBAL_PADDING)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    with torch.no_grad():
        out = model(batch)

    x_encoded = captured["x_encoded"]
    print(f"\nFull model output shape: {tuple(out.shape)}")
    print(f"Encoder hook output shape: {tuple(x_encoded.shape)}  [batch, n_global_mu, n_local_mu, embed_dim]")
    print(
        f"global_shape_mu should be "
        f"({n_lats_px // p['mask_unit_size_px'][0]}, {n_lons_px // p['mask_unit_size_px'][1]}) "
        f"-> {x_encoded.shape[1]} matches expectation: "
        f"{x_encoded.shape[1] == (n_lats_px // p['mask_unit_size_px'][0]) * (n_lons_px // p['mask_unit_size_px'][1])}"
    )
    print("\nSample validation passed.")


if __name__ == "__main__":
    main()
