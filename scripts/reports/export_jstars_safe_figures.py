import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_b1_report_assets import plot_threshold_sweep
from make_final_report_assets import (
    binary_from_prob_map as final_binary_from_prob_map,
    build_dataset as build_final_dataset,
    build_loader as build_final_loader,
    build_overlay as build_final_overlay,
    collect_prob_maps as collect_final_prob_maps,
    load_checkpoint_model as load_final_checkpoint_model,
    resize_raw_pair as resize_final_raw_pair,
    sample_metrics as final_sample_metrics,
)
from make_kaggle_progression_figure import (
    DEFAULT_SAMPLE_IDS as QUAL_PROGRESS_DEFAULT_SAMPLE_IDS,
    StageSpec as QualStageSpec,
    build_dataset as build_qual_dataset,
    build_loader as build_qual_loader,
    compute_stage_prob_maps,
    build_case_rows,
    render_progression_figure,
    resolve_sample_indices,
    write_metrics_csv,
    write_summary,
)
from paper_figures import (
    EXTERNAL_SAM799_DEFAULT_STAGE_ORDER,
    SelectedSample,
    build_b2_supervision_rows,
    load_dataset_records,
    render_b2_supervision_scaling,
    render_dataset_overview,
    render_external_sam799_single_case,
)


LINE_DPI = 600
COLOR_DPI = 400
SINGLE_COLUMN_COLOR_MIN_WIDTH = 1400
DOUBLE_COLUMN_COLOR_MIN_WIDTH = 2800
SINGLE_COLUMN_LINE_MIN_WIDTH = 2100
DOUBLE_COLUMN_LINE_MIN_WIDTH = 4200
SOURCE_ONLY_ORIGINAL_SAMPLE_IDS = ("slide1290", "slide1434", "slide1089")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a JSTARS-safe figure bundle with high-resolution re-renders "
            "for the main paper-facing plots and qualitative figures."
        )
    )
    parser.add_argument("--output-dir", default="results/report_assets/jstars_safe")
    parser.add_argument("--copy-dir", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_threshold_sweep_rows(csv_path: str | Path) -> list[dict]:
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                "model": row["model"],
                "threshold": float(row["threshold"]),
                "iou": float(row["iou"]),
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }
        )
    return parsed_rows


def load_selected_samples(csv_path: str | Path) -> dict[str, list[SelectedSample]]:
    selections: dict[str, list[SelectedSample]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    for row in rows:
        selections.setdefault(row["dataset_key"], []).append(
            SelectedSample(
                dataset_key=row["dataset_key"],
                dataset_label=row["dataset_label"],
                category=row["category"],
                sample_id=row["sample_id"],
                image_rel=row["image_rel"],
                mask_rel=row["mask_rel"],
                width=int(row["width"]),
                height=int(row["height"]),
                fg_ratio=float(row["fg_ratio_pct"]) / 100.0,
                split_file=row["split_file"],
            )
        )
    return selections


def render_seed_sweep(csv_path: str | Path, output_path: str | Path, dpi: int = LINE_DPI) -> Path:
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "training_group": row["training_group"],
                    "seed": int(row["seed"]),
                    "experiment_name": row["experiment_name"],
                    "iou": float(row["iou"]),
                    "f1": float(row["f1"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    group_order = ["Matched random background", "Mined TS-bank"]
    group_colors = {
        "Matched random background": "#2563eb",
        "Mined TS-bank": "#d97706",
    }
    x_positions = {group: idx for idx, group in enumerate(group_order)}

    metrics = [("iou", "Validation IoU"), ("f1", "Validation F1")]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7), sharex=False)

    by_seed: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["training_group"]] = row

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for seed in sorted(by_seed):
            seed_rows = by_seed[seed]
            if not all(group in seed_rows for group in group_order):
                continue
            xs = [x_positions[group] for group in group_order]
            ys = [seed_rows[group][metric_key] for group in group_order]
            ax.plot(
                xs,
                ys,
                color="#9ca3af",
                linewidth=1.35,
                marker="o",
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.2,
                alpha=0.9,
            )

        for group in group_order:
            values = np.asarray(
                [row[metric_key] for row in rows if row["training_group"] == group],
                dtype=np.float64,
            )
            x = x_positions[group]
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ax.errorbar(
                [x],
                [mean_value],
                yerr=[std_value],
                fmt="o",
                markersize=9.5,
                color=group_colors[group],
                elinewidth=2.0,
                capsize=6,
                capthick=2.0,
                markeredgewidth=1.5,
                zorder=5,
            )
            x_text = x - 0.18 if group == group_order[0] else x - 0.18
            ax.text(
                x_text,
                mean_value + (0.008 if metric_key == "iou" else 0.008),
                f"{mean_value:.3f} +/- {std_value:.3f}",
                color=group_colors[group],
                fontsize=10.5,
                ha="left",
                va="center",
            )

        ax.set_title(metric_title, fontsize=16)
        ax.set_ylabel(metric_title, fontsize=12)
        ax.set_xticks([0, 1], ["Matched\nrandom background", "Mined\nTS-bank"])
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Matched random background vs mined TS-bank on UAV validation", fontsize=17)
    proxy_line = plt.Line2D(
        [0],
        [0],
        color="#9ca3af",
        linewidth=1.35,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=6.2,
        label="Paired seed reruns",
    )
    proxy_point = plt.Line2D(
        [0],
        [0],
        color="black",
        linewidth=0,
        marker="o",
        markersize=8.5,
        label="Group mean +/- SD",
    )
    axes[0].legend(handles=[proxy_line, proxy_point], frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_qualitative_progression(output_dir: Path, device_arg: str) -> list[Path]:
    stage_specs = [
        QualStageSpec(
            key="source_only",
            title="Source-only\nSegFormer-B2",
            caption_label="Source-only SegFormer-B2",
            model_name="SegFormer-B2",
            checkpoint_path="checkpoints/segformer_b2_plain_360.pth",
            threshold=0.5,
        ),
        QualStageSpec(
            key="advent",
            title="ADVENT-style\nDeepLabV3+",
            caption_label="ADVENT-style DeepLabV3+",
            model_name="DeepLabV3Plus",
            checkpoint_path="checkpoints/advent_deeplabv3plus_crack500_to_uav.pth",
            threshold=0.9,
        ),
        QualStageSpec(
            key="b1_selected",
            title="Selected B1\nSegFormer-B2",
            caption_label="Selected B1 SegFormer-B2",
            model_name="SegFormer-B2",
            checkpoint_path="checkpoints/segformer_b2_b1_tsbank_thr080_mean082.pth",
            threshold=0.6,
        ),
        QualStageSpec(
            key="b2_fs10",
            title="B2 fs10\nSegFormer-B2",
            caption_label="B2 fs10 SegFormer-B2",
            model_name="SegFormer-B2",
            checkpoint_path="checkpoints/segformer_b2_b2_fs10_seed42.pth",
            threshold=0.5,
        ),
    ]

    dataset = build_qual_dataset(root="UAV_Crack_Segmentation_Kaggle", split="test", img_size=360)
    loader = build_qual_loader(dataset=dataset, batch_size=8, num_workers=0)
    sample_indices = resolve_sample_indices(dataset=dataset, sample_ids=list(QUAL_PROGRESS_DEFAULT_SAMPLE_IDS))
    if device_arg == "auto":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch

        device = torch.device(device_arg)
    stage_prob_maps = compute_stage_prob_maps(stage_specs=stage_specs, loader=loader, device=device)
    rows = build_case_rows(
        dataset=dataset,
        sample_indices=sample_indices,
        stage_specs=stage_specs,
        stage_prob_maps=stage_prob_maps,
    )

    image_path = output_dir / "qualitative_progression_test.png"
    metrics_path = output_dir / "qualitative_progression_metrics.csv"
    summary_path = output_dir / "qualitative_progression_summary.md"
    render_progression_figure(
        output_path=image_path,
        rows=rows,
        stage_specs=stage_specs,
        dpi=COLOR_DPI,
    )
    write_metrics_csv(output_path=metrics_path, rows=rows, stage_specs=stage_specs)
    write_summary(output_path=summary_path, rows=rows, stage_specs=stage_specs)

    final_alias = output_dir / "qualitative_progression_test_final.png"
    shutil.copy2(image_path, final_alias)
    return [image_path, metrics_path, summary_path, final_alias]


def export_original_sourceonly(output_dir: Path, device_arg: str) -> list[Path]:
    dataset = build_final_dataset(root="UAV_Crack_Segmentation_Kaggle", split="test", img_size=360)
    loader = build_final_loader(dataset=dataset, batch_size=8, num_workers=0)
    if device_arg == "auto":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch

        device = torch.device(device_arg)

    model = load_final_checkpoint_model(
        model_name="segformer-b2",
        checkpoint_path="checkpoints/segformer_b2_plain_360.pth",
        device=device,
    )
    stage_prob_maps = collect_final_prob_maps(model=model, loader=loader, device=device)
    del model
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    sample_indices = resolve_sample_indices(dataset=dataset, sample_ids=list(SOURCE_ONLY_ORIGINAL_SAMPLE_IDS))
    cases = []
    for idx in sample_indices:
        sample_id = dataset.samples[idx][0].stem
        _, gt_mask = resize_final_raw_pair(dataset, idx, dataset.img_size)
        source_pred = final_binary_from_prob_map(stage_prob_maps[idx], 0.5)
        source_stats = final_sample_metrics(source_pred, gt_mask)
        cases.append(
            {
                "idx": idx,
                "sample_id": sample_id,
                "source_stats": source_stats,
            }
        )

    output_path = output_dir / "qualitative_sourceonly_test_original.png"
    fig, axes = plt.subplots(
        len(cases),
        3,
        figsize=(9.6, 3.22 * len(cases)),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for col_idx, title in enumerate(("Input", "Ground Truth", "Source-only prediction (thr. = 0.5)")):
        axes[0, col_idx].set_title(title)

    for row_idx, case in enumerate(cases):
        idx = case["idx"]
        image, gt_mask = resize_final_raw_pair(dataset, idx, dataset.img_size)
        gt_rgb = np.repeat((gt_mask * 255)[:, :, None], 3, axis=2)
        source_pred = final_binary_from_prob_map(stage_prob_maps[idx], 0.5)
        source_overlay = build_final_overlay(image, source_pred, gt_mask)
        panels = [image, gt_rgb, source_overlay]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(case["sample_id"], rotation=0, labelpad=46, fontsize=10, va="center")
            if col_idx == 2:
                stats = case["source_stats"]
                ax.text(
                    0.5,
                    -0.017,
                    f"IoU {stats['iou']:.3f} | P {stats['precision']:.3f} | R {stats['recall']:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                    clip_on=False,
                )

    fig.text(
        0.5,
        0.001,
        "Green = true positive, red = false positive, yellow = false negative",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.955, bottom=0.048, hspace=0.05, wspace=0.05)
    fig.savefig(output_path, dpi=COLOR_DPI, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)
    return [output_path]


def export_threshold_sweep(output_dir: Path) -> list[Path]:
    sweep_rows = load_threshold_sweep_rows("results/report_assets/b1_holdout/threshold_sweep_val.csv")
    plot_threshold_sweep(output_dir=output_dir, sweep_rows=sweep_rows, dpi=LINE_DPI)
    return [output_dir / "threshold_sweep_val.png"]


def export_dataset_overview(output_dir: Path) -> list[Path]:
    records_by_dataset = load_dataset_records()
    selections = load_selected_samples("results/report_assets/paper_figures/dataset_overview_selected_samples.csv")
    output_path = output_dir / "dataset_overview_representative_360.png"
    render_dataset_overview(
        records_by_dataset=records_by_dataset,
        selections=selections,
        output_path=output_path,
        preprocess_square_size=360,
        dpi=COLOR_DPI,
    )
    return [output_path]


def export_b2_recovery(output_dir: Path) -> list[Path]:
    rows = build_b2_supervision_rows("results/experiments.csv")
    output_path = output_dir / "b2_supervision_scaling_holdout_reflines.png"
    render_b2_supervision_scaling(rows=rows, output_path=output_path, dpi=LINE_DPI)
    return [output_path]


def export_sam799_single_case(output_dir: Path) -> list[Path]:
    output_path = output_dir / "sam799_external_single_case_dji0069_source_b1_b2fs10.png"
    render_external_sam799_single_case(
        dataset_root="SAM799_CVAT",
        external_results_root="results/external_sam799_cvat_patchwise",
        per_image_csv="results/external_sam799_cvat_patchwise/per_image_metrics.csv",
        output_path=output_path,
        image_id="DJI_0069_JPG.rf.WxOwzvxUj9I8ZXZzzVnv",
        stage_order=EXTERNAL_SAM799_DEFAULT_STAGE_ORDER,
        dpi=COLOR_DPI,
    )
    return [output_path]


def export_seed_sweep(output_dir: Path) -> list[Path]:
    output_path = output_dir / "segformer_random_bg_vs_mined_seed_sweep.png"
    render_seed_sweep(
        csv_path="results/report_assets/paper_figures/b1_mechanism/segformer_random_bg_vs_mined_seed_sweep.csv",
        output_path=output_path,
        dpi=LINE_DPI,
    )
    return [output_path]


def inspect_image(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
        dpi = image.info.get("dpi", "")
    is_line = any(
        token in path.name
        for token in (
            "threshold_sweep",
            "seed_sweep",
            "supervision_scaling",
            "reflines",
        )
    )
    min_width_single = SINGLE_COLUMN_LINE_MIN_WIDTH if is_line else SINGLE_COLUMN_COLOR_MIN_WIDTH
    min_width_double = DOUBLE_COLUMN_LINE_MIN_WIDTH if is_line else DOUBLE_COLUMN_COLOR_MIN_WIDTH
    return {
        "file_name": path.name,
        "path": str(path),
        "kind": "line" if is_line else "color",
        "width_px": width,
        "height_px": height,
        "embedded_dpi": dpi,
        "single_col_min_width_px": min_width_single,
        "double_col_min_width_px": min_width_double,
        "passes_single_col": width >= min_width_single,
        "passes_double_col": width >= min_width_double,
    }


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> list[Path]:
    csv_path = output_dir / "jstars_safe_manifest.csv"
    md_path = output_dir / "README.md"
    fieldnames = [
        "file_name",
        "kind",
        "width_px",
        "height_px",
        "embedded_dpi",
        "single_col_min_width_px",
        "double_col_min_width_px",
        "passes_single_col",
        "passes_double_col",
        "path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# JSTARS-safe figure bundle",
        "",
        "- Color figures were re-exported at `400 dpi`.",
        "- Line/curve figures were re-exported at `600 dpi`.",
        "- JSTARS pixel-width checks use: single-column color `1400 px`, double-column color `2800 px`, single-column line `2100 px`, double-column line `4200 px`.",
        "",
        "| File | Kind | Width | Height | Double-column pass |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['file_name']}` | `{row['kind']}` | `{row['width_px']}` | "
            f"`{row['height_px']}` | `{str(row['passes_double_col']).lower()}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def copy_exports(paths: list[Path], copy_dir: Path):
    copy_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, copy_dir / path.name)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    exported_paths.extend(export_b2_recovery(output_dir))
    exported_paths.extend(export_dataset_overview(output_dir))
    exported_paths.extend(export_qualitative_progression(output_dir, device_arg=args.device))
    exported_paths.extend(export_original_sourceonly(output_dir, device_arg=args.device))
    exported_paths.extend(export_sam799_single_case(output_dir))
    exported_paths.extend(export_seed_sweep(output_dir))
    exported_paths.extend(export_threshold_sweep(output_dir))

    manifest_rows = [inspect_image(path) for path in exported_paths if path.suffix.lower() == ".png"]
    exported_paths.extend(write_manifest(output_dir=output_dir, rows=manifest_rows))

    if args.copy_dir:
        copy_exports(exported_paths, Path(args.copy_dir).resolve())

    print(f"Wrote JSTARS-safe figure bundle to: {output_dir}")
    for row in manifest_rows:
        print(
            f"- {row['file_name']}: {row['width_px']}x{row['height_px']} "
            f"(double-column pass={str(row['passes_double_col']).lower()})"
        )


if __name__ == "__main__":
    main()
