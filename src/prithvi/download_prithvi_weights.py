"""Download prithvi.wxc.2300m.v1 weights and climatology into scratch.

Uses huggingface_hub so HF_HOME should already be redirected to scratch
(set by the calling sbatch script) before this runs.

This repo ships climatology as thousands of small per-day files, which is
enough traffic to hit Hugging Face's unauthenticated rate limit (429 Too Many
Requests) partway through. Two mitigations:
  1. Set HF_TOKEN (a free HF account's access token) before running --
     authenticated requests get a much higher rate limit.
  2. This script retries on 429/connection errors with backoff, and
     snapshot_download with local_dir is resumable -- already-downloaded
     files aren't re-fetched, so a retry (or just re-running the sbatch job)
     picks up where it left off.
"""
import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # older huggingface_hub versions
    from huggingface_hub.utils import HfHubHTTPError

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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent download threads (default 2 -- kept low since bursts of parallel requests seem to trip the rate limit even with HF_TOKEN set).",
    )
    parser.add_argument("--max-retries", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = None
    for attempt in range(1, args.max_retries + 1):
        try:
            snapshot_dir = snapshot_download(
                repo_id=args.repo_id,
                local_dir=args.out_dir,
                max_workers=args.max_workers,
            )
            break
        except (HfHubHTTPError, ConnectionError) as e:
            wait = min(30 * attempt, 600)
            print(
                f"Attempt {attempt}/{args.max_retries} failed ({e}); "
                f"retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)
    if snapshot_dir is None:
        raise SystemExit(
            f"Gave up after {args.max_retries} attempts. Set HF_TOKEN (free HF "
            "account access token) to raise the rate limit, then re-run -- "
            "already-downloaded files won't be re-fetched."
        )

    print(f"Downloaded {args.repo_id} to: {snapshot_dir}")

    print("\nFiles:")
    for p in sorted(Path(snapshot_dir).rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(snapshot_dir)}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
