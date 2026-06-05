import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MINE_SCRIPT = REPO_ROOT / "scripts" / "banks" / "mine_uav_hard_negatives.py"
FILTER_SCRIPT = REPO_ROOT / "scripts" / "banks" / "export_auto_filtered_hard_negative_bank.py"
DEFAULT_CONFIG = Path(__file__).resolve().with_name("b1_candidate_bank_parameters.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Public-facing wrapper for the B1 hard-negative mining and filtering pipeline. "
            "It reads preset parameters from YAML and dispatches to the repo's lower-level bank scripts."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML file that defines mining_presets and filter_presets.",
    )
    parser.add_argument(
        "--mode",
        choices=("mine", "filter", "mine-and-filter"),
        required=True,
        help="Which stage of the B1 bank workflow to run.",
    )
    parser.add_argument(
        "--mine-preset",
        default=None,
        help="Preset key under mining_presets.",
    )
    parser.add_argument(
        "--filter-preset",
        default=None,
        help="Preset key under filter_presets.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping at the top level of {config_path}")
    return payload


def encode_args(mapping: dict[str, object]) -> list[str]:
    cli_args: list[str] = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            cli_args.append(flag)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def run_script(script_path: Path, config: dict[str, object]):
    cmd = [sys.executable, str(script_path), *encode_args(config)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    payload = load_config(args.config)
    defaults = dict(payload.get("defaults", {}))
    mining_presets = dict(payload.get("mining_presets", {}))
    filter_presets = dict(payload.get("filter_presets", {}))

    if args.mode in {"mine", "mine-and-filter"}:
        if not args.mine_preset:
            raise ValueError("--mine-preset is required for mode=mine or mine-and-filter")
        if args.mine_preset not in mining_presets:
            raise KeyError(f"Unknown mining preset: {args.mine_preset}")
        mining_config = dict(defaults)
        mining_config.update(mining_presets[args.mine_preset])
        run_script(MINE_SCRIPT, mining_config)
    else:
        mining_config = {}

    if args.mode in {"filter", "mine-and-filter"}:
        if not args.filter_preset:
            raise ValueError("--filter-preset is required for mode=filter or mine-and-filter")
        if args.filter_preset not in filter_presets:
            raise KeyError(f"Unknown filter preset: {args.filter_preset}")
        filter_config = dict(filter_presets[args.filter_preset])
        if args.mode == "mine-and-filter" and "bank_root" not in filter_config:
            filter_config["bank_root"] = mining_config.get("output_root")
        run_script(FILTER_SCRIPT, filter_config)


if __name__ == "__main__":
    main()
