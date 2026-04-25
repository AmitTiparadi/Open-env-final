"""HF Jobs bootstrap runner for Incident Commander training.

HF Jobs `uv run <script-url>` downloads only that script, not the whole Space
repository. This wrapper downloads the Space snapshot first, installs the
project, and then executes one training stage.

Examples:
    hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
      https://huggingface.co/spaces/AmitTiparadi/Open-env-finals/resolve/main/training/run_hf_job.py \
      --stage pretrain -- --dry-run

    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN \
      https://huggingface.co/spaces/AmitTiparadi/Open-env-finals/resolve/main/training/run_hf_job.py \
      --stage pretrain -- --push-to-hub --hub-model-id AmitTiparadi/incident-commander-pretrain
"""

# /// script
# dependencies = ["huggingface_hub>=1.0.0"]
# ///

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


SPACE_ID = "AmitTiparadi/Open-env-finals"
STAGE_TO_SCRIPT = {
    "prepare": "training/prepare_data.py",
    "pretrain": "training/train_pretrain.py",
    "sft": "training/train_sft.py",
    "grpo": "training/train_grpo.py",
}


def run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def install_project(repo_path: Path, extras_target: str) -> None:
    uv = shutil.which("uv")
    if uv:
        run(
            [uv, "pip", "install", "--python", sys.executable, "-e", extras_target],
            cwd=repo_path,
        )
        return
    run([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=repo_path)
    run([sys.executable, "-m", "pip", "install", "-e", extras_target], cwd=repo_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.getenv("SPACE_REPO_ID", SPACE_ID))
    parser.add_argument("--revision", default=os.getenv("SPACE_REVISION", "main"))
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_TO_SCRIPT),
        default="pretrain",
        help="Training stage to run.",
    )
    args, stage_args = parser.parse_known_args()
    stage_args = list(stage_args)
    if stage_args and stage_args[0] == "--":
        stage_args = stage_args[1:]

    print(
        f"Downloading Space repo {args.repo_id}@{args.revision} into the HF Job...",
        flush=True,
    )
    repo_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="space",
            revision=args.revision,
            local_dir="/tmp/incident-commander-space",
        )
    )

    is_dry_run = "--dry-run" in stage_args or args.stage == "prepare"
    extras_target = ".[training]" if not is_dry_run else "."

    print(
        f"Installing project {'with training extras' if not is_dry_run else 'without training extras'}...",
        flush=True,
    )
    install_project(repo_path, extras_target)

    if args.stage != "prepare":
        data_path = repo_path / "data" / "pretrain_corpus.jsonl"
        if not data_path.exists():
            print("Training data missing; generating datasets first.", flush=True)
            run([sys.executable, STAGE_TO_SCRIPT["prepare"]], cwd=repo_path)

    selected_script = STAGE_TO_SCRIPT[args.stage]
    run([sys.executable, selected_script, *stage_args], cwd=repo_path)


if __name__ == "__main__":
    main()
