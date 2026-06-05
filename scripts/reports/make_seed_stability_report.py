import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    stage: str
    threshold: float
    experiment_name_template: str


SETTING_SPECS = [
    SettingSpec(
        key="source_only",
        label="Source-only @ 0.5",
        stage="Source-only",
        threshold=0.5,
        experiment_name_template="segformer_b2_plain_360_seed{seed:03d}_uav_holdout_raw_test",
    ),
    SettingSpec(
        key="b1_promoted",
        label="B1 promoted TS-bank @ 0.6",
        stage="B1",
        threshold=0.6,
        experiment_name_template="segformer_b2_b1_tsbank_thr080_mean082_seed{seed:03d}_test_thr060",
    ),
    SettingSpec(
        key="b2_fs05",
        label="B2 fs05 @ 0.5",
        stage="B2",
        threshold=0.5,
        experiment_name_template="segformer_b2_b2_fs05_datasplit042_seed{seed:03d}_test",
    ),
    SettingSpec(
        key="b2_fs10",
        label="B2 fs10 @ 0.5",
        stage="B2",
        threshold=0.5,
        experiment_name_template="segformer_b2_b2_fs10_datasplit042_seed{seed:03d}_test",
    ),
    SettingSpec(
        key="b2_fs20",
        label="B2 fs20 @ 0.5",
        stage="B2",
        threshold=0.5,
        experiment_name_template="segformer_b2_b2_fs20_datasplit042_seed{seed:03d}_test",
    ),
]

METRIC_KEYS = ("metric_iou", "metric_f1", "metric_precision", "metric_recall")
METRIC_LABELS = {
    "metric_iou": "iou",
    "metric_f1": "f1",
    "metric_precision": "precision",
    "metric_recall": "recall",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the SegFormer-B2 seed-stability reruns into "
            "paper-facing CSV and Markdown tables."
        )
    )
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument("--output-dir", default="results/report_assets/seed_stability")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42])
    parser.add_argument("--stage-keys", nargs="+", default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def load_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results CSV: {results_csv}")
    with results_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        experiment_name = row.get("experiment_name", "")
        if experiment_name:
            lookup[experiment_name] = row
    return lookup


def filter_specs(stage_keys: list[str] | None) -> list[SettingSpec]:
    if not stage_keys:
        return list(SETTING_SPECS)
    wanted = set(stage_keys)
    specs = [spec for spec in SETTING_SPECS if spec.key in wanted]
    missing = wanted.difference({spec.key for spec in specs})
    if missing:
        raise ValueError(f"Unknown --stage-keys requested: {sorted(missing)}")
    return specs


def collect_per_run_rows(
    lookup: dict[str, dict[str, str]],
    setting_specs: list[SettingSpec],
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    per_run_rows: list[dict[str, float | int | str]] = []
    for spec in setting_specs:
        for seed in seeds:
            experiment_name = spec.experiment_name_template.format(seed=seed)
            if experiment_name not in lookup:
                raise KeyError(
                    "Missing seed-stability result row for "
                    f"`{experiment_name}` in results/experiments.csv"
                )
            row = lookup[experiment_name]
            if row.get("script") != "test.py" or row.get("stage") != "test" or row.get("split") != "test":
                raise ValueError(
                    f"Expected `{experiment_name}` to be a formal `test.py` hold-out row, "
                    f"got script={row.get('script')} stage={row.get('stage')} split={row.get('split')}"
                )
            per_run_rows.append(
                {
                    "setting_key": spec.key,
                    "setting_label": spec.label,
                    "stage": spec.stage,
                    "seed": seed,
                    "threshold": spec.threshold,
                    "experiment_name": experiment_name,
                    "checkpoint_path": row.get("checkpoint_path", ""),
                    "iou": float(row["metric_iou"]),
                    "f1": float(row["metric_f1"]),
                    "precision": float(row["metric_precision"]),
                    "recall": float(row["metric_recall"]),
                }
            )
    return per_run_rows


def summarize_rows(
    per_run_rows: list[dict[str, float | int | str]],
    setting_specs: list[SettingSpec],
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    summary_rows: list[dict[str, float | int | str]] = []
    for spec in setting_specs:
        matching = [row for row in per_run_rows if row["setting_key"] == spec.key]
        if len(matching) != len(seeds):
            raise ValueError(
                f"Setting `{spec.key}` expected {len(seeds)} rows, found {len(matching)}"
            )
        summary: dict[str, float | int | str] = {
            "setting_key": spec.key,
            "setting_label": spec.label,
            "stage": spec.stage,
            "threshold": spec.threshold,
            "seed_count": len(matching),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric_key, metric_label in METRIC_LABELS.items():
            values = [float(row[metric_label]) for row in matching]
            summary[f"{metric_label}_mean"] = mean(values)
            summary[f"{metric_label}_std"] = sample_std(values)
            summary[f"{metric_label}_values"] = ", ".join(f"{value:.4f}" for value in values)
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]):
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def write_summary_md(
    path: Path,
    summary_rows: list[dict[str, float | int | str]],
    seeds: list[int],
):
    lines = [
        "# Seed Stability Summary",
        "",
        "This report summarizes fixed-hold-out `UAV test` reruns for selected `SegFormer-B2` settings.",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "- Test split: `UAV_Crack_Segmentation_Kaggle/test` (`63` images)",
        "- Purpose: quantify training stochasticity on the same fixed test split",
        "",
        "## Mean ± Std",
        "",
        "| Setting | Stage | IoU | F1 | Precision | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['setting_label']} | "
            f"{row['stage']} | "
            f"{format_mean_std(float(row['iou_mean']), float(row['iou_std']))} | "
            f"{format_mean_std(float(row['f1_mean']), float(row['f1_std']))} | "
            f"{format_mean_std(float(row['precision_mean']), float(row['precision_std']))} | "
            f"{format_mean_std(float(row['recall_mean']), float(row['recall_std']))} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed IoU",
            "",
        ]
    )
    for row in summary_rows:
        lines.append(f"- `{row['setting_label']}`: {row['iou_values']}")

    lines.extend(
        [
            "",
            "## Suggested Paper Text",
            "",
            (
                "To assess training stochasticity, we repeated the selected settings with "
                f"`{len(seeds)}` random seeds for `SegFormer-B2` and report mean ± standard "
                "deviation on the fixed `UAV test` split. The cross-seed variance can then be "
                "compared directly against the between-setting gaps."
            ),
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results_csv)
    lookup = build_lookup(rows)
    setting_specs = filter_specs(args.stage_keys)
    per_run_rows = collect_per_run_rows(lookup=lookup, setting_specs=setting_specs, seeds=args.seeds)
    summary_rows = summarize_rows(
        per_run_rows=per_run_rows,
        setting_specs=setting_specs,
        seeds=args.seeds,
    )

    write_csv(output_dir / "per_run.csv", per_run_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_summary_md(output_dir / "summary.md", summary_rows=summary_rows, seeds=args.seeds)

    print(f"Wrote seed-stability assets to: {output_dir}")


if __name__ == "__main__":
    main()
