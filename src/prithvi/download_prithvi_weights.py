"""Download prithvi.wxc.2300m.v1 weights and climatology into scratch.

Uses huggingface_hub so HF_HOME should already be redirected to scratch
(set by the calling sbatch script) before this runs.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "Prithvi-WxC/prithvi.wxc.2300m.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to place the downloaded snapshot symlink/copy in.",
    )
    parser.add_argument(
        "--repo-id",
        default=MODEL_REPO,
        help=f"Hugging Face repo id (default: {MODEL_REPO}).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshot_download(repo_id=args.repo_id, local_dir=args.out_dir)
    print(f"Downloaded {args.repo_id} to: {snapshot_dir}")

    print("\nFiles:")
    for p in sorted(Path(snapshot_dir).rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(snapshot_dir)}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
