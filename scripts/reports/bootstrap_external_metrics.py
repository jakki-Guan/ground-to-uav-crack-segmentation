import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap image-level confidence intervals for patchwise SAM799 external "
            "evaluation results using per-image TP/FP/FN counts."
        )
    )
    parser.add_argument(
        "--per-image-metrics",
        default="results/external_sam799_cvat_patchwise/per_image_metrics.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="results/external_sam799_cvat_patchwise/bootstrap_ci.csv",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Per-image metrics CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full_like(numerator, fill_value=np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    return {
        "iou": safe_divide(tp, tp + fp + fn),
        "f1": safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
    }


def bootstrap_stage_rows(
    stage_rows: list[dict[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sample_count = len(stage_rows)
    indices = rng.integers(0, sample_count, size=(bootstrap_samples, sample_count))

    tp = np.asarray([int(row["tp"]) for row in stage_rows], dtype=np.int64)
    fp = np.asarray([int(row["fp"]) for row in stage_rows], dtype=np.int64)
    fn = np.asarray([int(row["fn"]) for row in stage_rows], dtype=np.int64)

    tp_sum = tp[indices].sum(axis=1, dtype=np.int64)
    fp_sum = fp[indices].sum(axis=1, dtype=np.int64)
    fn_sum = fn[indices].sum(axis=1, dtype=np.int64)
    metric_arrays = metrics_from_counts(tp=tp_sum, fp=fp_sum, fn=fn_sum)

    point_metrics = metrics_from_counts(
        tp=np.asarray([tp.sum()], dtype=np.int64),
        fp=np.asarray([fp.sum()], dtype=np.int64),
        fn=np.asarray([fn.sum()], dtype=np.int64),
    )

    first = stage_rows[0]
    row: dict[str, object] = {
        "stage_key": first["stage_key"],
        "stage_label": first["stage_label"],
        "checkpoint_path": first["checkpoint_path"],
        "threshold": float(first["threshold"]),
        "num_images": sample_count,
        "num_bootstrap": bootstrap_samples,
    }
    for metric_name, values in metric_arrays.items():
        row[metric_name] = float(point_metrics[metric_name][0])
        row[f"{metric_name}_ci_low"] = float(np.nanquantile(values, 0.025))
        row[f"{metric_name}_ci_high"] = float(np.nanquantile(values, 0.975))
    return row


def main():
    args = parse_args()
    per_image_metrics_path = Path(args.per_image_metrics).resolve()
    rows = load_rows(per_image_metrics_path)

    grouped_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["stage_key"], row["checkpoint_path"], row["threshold"])].append(row)

    output_rows: list[dict[str, object]] = []
    for stage_index, group_key in enumerate(sorted(grouped_rows.keys()), start=1):
        stage_rows = grouped_rows[group_key]
        output_rows.append(
            bootstrap_stage_rows(
                stage_rows=stage_rows,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + stage_index,
            )
        )

    output_csv = Path(args.output_csv).resolve()
    write_csv(output_csv, output_rows)
    print(f"Loaded per-image metrics from: {per_image_metrics_path}")
    print(f"Wrote bootstrap CIs to: {output_csv}")


if __name__ == "__main__":
    main()
