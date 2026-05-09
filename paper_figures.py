from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    root: str
    split_files: tuple[str, ...]


@dataclass(frozen=True)
class DatasetRecord:
    dataset_key: str
    dataset_label: str
    root: str
    split_file: str
    sample_id: str
    image_rel: str
    mask_rel: str
    width: int
    height: int
    fg_ratio: float
    contrast: float


@dataclass(frozen=True)
class SelectedSample:
    dataset_key: str
    dataset_label: str
    category: str
    sample_id: str
    image_rel: str
    mask_rel: str
    width: int
    height: int
    fg_ratio: float
    split_file: str


DEFAULT_CATEGORY_SPECS: tuple[tuple[str, float], ...] = (
    ("Sparse", 0.10),
    ("Typical", 0.50),
    ("Dense", 0.90),
    ("Very dense", 0.98),
)


def default_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="crack500",
            label="Crack500",
            root="CRACK500",
            split_files=("train.txt", "val.txt", "test.txt"),
        ),
        DatasetSpec(
            key="uav_kaggle",
            label="Kaggle UAV crack dataset",
            root="UAV_Crack_Segmentation_Kaggle",
            split_files=("crossdomain_all.txt",),
        ),
        DatasetSpec(
            key="pavecrack1300",
            label="PaveCrack1300",
            root="PaveCrack1300",
            split_files=("crossdomain_all.txt",),
        ),
    ]


def resolve_dataset_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path
    return root_path


def _read_split_records(spec: DatasetSpec) -> list[DatasetRecord]:
    root = resolve_dataset_root(spec.root)
    records: list[DatasetRecord] = []
    seen_pairs: set[tuple[str, str]] = set()

    for split_file in spec.split_files:
        split_path = root / split_file
        with split_path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                image_rel, mask_rel = stripped.split()
                pair_key = (image_rel, mask_rel)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                image_path = root / image_rel
                mask_path = root / mask_rel
                image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)
                height, width = mask.shape
                records.append(
                    DatasetRecord(
                        dataset_key=spec.key,
                        dataset_label=spec.label,
                        root=str(root),
                        split_file=split_file,
                        sample_id=Path(image_rel).stem,
                        image_rel=image_rel,
                        mask_rel=mask_rel,
                        width=int(width),
                        height=int(height),
                        fg_ratio=float(mask.mean()),
                        contrast=float(image.std()),
                    )
                )

    records.sort(key=lambda row: (row.fg_ratio, row.sample_id))
    return records


def load_dataset_records(specs: list[DatasetSpec] | None = None) -> dict[str, list[DatasetRecord]]:
    specs = specs or default_dataset_specs()
    return {spec.key: _read_split_records(spec) for spec in specs}


def summarize_dataset_records(records_by_dataset: dict[str, list[DatasetRecord]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for dataset_key, records in records_by_dataset.items():
        if not records:
            continue
        fg_values = np.array([row.fg_ratio for row in records], dtype=np.float64)
        size_values = sorted({(row.width, row.height) for row in records})
        summaries.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": records[0].dataset_label,
                "num_samples": len(records),
                "native_sizes": ", ".join(f"{w}x{h}" for w, h in size_values),
                "fg_mean_pct": float(fg_values.mean() * 100.0),
                "fg_median_pct": float(np.median(fg_values) * 100.0),
                "fg_min_pct": float(fg_values.min() * 100.0),
                "fg_max_pct": float(fg_values.max() * 100.0),
            }
        )
    return summaries


def _choose_rank_representative(
    sorted_records: list[DatasetRecord],
    target_index: int,
    used_sample_ids: set[str],
    neighborhood: int,
) -> DatasetRecord:
    left = max(0, target_index - neighborhood)
    right = min(len(sorted_records), target_index + neighborhood + 1)
    candidates = [
        (abs(idx - target_index), -sorted_records[idx].contrast, idx, sorted_records[idx])
        for idx in range(left, right)
        if sorted_records[idx].sample_id not in used_sample_ids
    ]
    if not candidates:
        candidates = [
            (abs(idx - target_index), -row.contrast, idx, row)
            for idx, row in enumerate(sorted_records)
            if row.sample_id not in used_sample_ids
        ]
    _, _, _, best_record = min(candidates)
    return best_record


def auto_select_representative_samples(
    records_by_dataset: dict[str, list[DatasetRecord]],
    category_specs: tuple[tuple[str, float], ...] = DEFAULT_CATEGORY_SPECS,
    neighborhood: int = 24,
    manual_overrides: dict[str, dict[str, str]] | None = None,
) -> dict[str, list[SelectedSample]]:
    manual_overrides = manual_overrides or {}
    selections: dict[str, list[SelectedSample]] = {}

    for dataset_key, records in records_by_dataset.items():
        records_by_id = {row.sample_id: row for row in records}
        chosen_rows: list[SelectedSample] = []
        used_sample_ids: set[str] = set()
        overrides_for_dataset = manual_overrides.get(dataset_key, {})

        for category_label, quantile in category_specs:
            override_sample_id = overrides_for_dataset.get(category_label)
            if override_sample_id:
                if override_sample_id not in records_by_id:
                    raise KeyError(
                        f"Manual override '{override_sample_id}' not found in dataset '{dataset_key}'."
                    )
                picked = records_by_id[override_sample_id]
            else:
                target_index = int(round((len(records) - 1) * quantile))
                picked = _choose_rank_representative(
                    sorted_records=records,
                    target_index=target_index,
                    used_sample_ids=used_sample_ids,
                    neighborhood=neighborhood,
                )

            used_sample_ids.add(picked.sample_id)
            chosen_rows.append(
                SelectedSample(
                    dataset_key=picked.dataset_key,
                    dataset_label=picked.dataset_label,
                    category=category_label,
                    sample_id=picked.sample_id,
                    image_rel=picked.image_rel,
                    mask_rel=picked.mask_rel,
                    width=picked.width,
                    height=picked.height,
                    fg_ratio=picked.fg_ratio,
                    split_file=picked.split_file,
                )
            )
        selections[dataset_key] = chosen_rows

    return selections


def select_random_samples(
    records_by_dataset: dict[str, list[DatasetRecord]],
    count: int = 4,
    seed: int = 42,
) -> dict[str, list[SelectedSample]]:
    rng = random.Random(seed)
    selections: dict[str, list[SelectedSample]] = {}
    for dataset_key, records in records_by_dataset.items():
        sample_count = min(count, len(records))
        picked_records = rng.sample(records, sample_count)
        picked_records.sort(key=lambda row: row.sample_id)
        selections[dataset_key] = [
            SelectedSample(
                dataset_key=row.dataset_key,
                dataset_label=row.dataset_label,
                category=f"Random {idx + 1}",
                sample_id=row.sample_id,
                image_rel=row.image_rel,
                mask_rel=row.mask_rel,
                width=row.width,
                height=row.height,
                fg_ratio=row.fg_ratio,
                split_file=row.split_file,
            )
            for idx, row in enumerate(picked_records)
        ]
    return selections


def _build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    crack_color = np.array([255, 92, 28], dtype=np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = 0.35 * overlay[mask_bool] + 0.65 * crack_color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _fit_to_canvas(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    fill_value: int = 255,
) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    scale = min(target_width / src_width, target_height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))

    pil_image = Image.fromarray(image)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    resized_np = np.array(resized, dtype=np.uint8)

    canvas = np.full((target_height, target_width, 3), fill_value, dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top: top + resized_height, left: left + resized_width] = resized_np
    return canvas


def _resize_image_and_mask_to_square(
    image: np.ndarray,
    mask: np.ndarray,
    square_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_resized = np.array(
        Image.fromarray(image).resize(
            (square_size, square_size),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    mask_resized = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (square_size, square_size),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    return image_resized, (mask_resized > 0).astype(np.uint8)


def _load_card(
    sample: SelectedSample,
    panel_width: int,
    panel_height: int,
    preprocess_square_size: int | None = None,
) -> np.ndarray:
    root = resolve_dataset_root(
        next(spec.root for spec in default_dataset_specs() if spec.key == sample.dataset_key)
    )
    image = np.array(Image.open(root / sample.image_rel).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(root / sample.mask_rel).convert("1"), dtype=np.uint8)
    if preprocess_square_size is not None:
        image, mask = _resize_image_and_mask_to_square(
            image=image,
            mask=mask,
            square_size=preprocess_square_size,
        )
    overlay = _build_overlay(image=image, mask=mask)
    native_card = np.vstack([image, overlay])
    return _fit_to_canvas(
        image=native_card,
        target_width=panel_width,
        target_height=panel_height,
    )


def flatten_selected_samples(selections: dict[str, list[SelectedSample]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_key, picked_rows in selections.items():
        for row in picked_rows:
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": row.dataset_label,
                    "category": row.category,
                    "sample_id": row.sample_id,
                    "image_rel": row.image_rel,
                    "mask_rel": row.mask_rel,
                    "split_file": row.split_file,
                    "width": row.width,
                    "height": row.height,
                    "fg_ratio_pct": row.fg_ratio * 100.0,
                }
            )
    return rows


def write_selection_csv(selections: dict[str, list[SelectedSample]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = flatten_selected_samples(selections)
    fieldnames = [
        "dataset_key",
        "dataset_label",
        "category",
        "sample_id",
        "image_rel",
        "mask_rel",
        "split_file",
        "width",
        "height",
        "fg_ratio_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def render_dataset_overview(
    records_by_dataset: dict[str, list[DatasetRecord]],
    selections: dict[str, list[SelectedSample]],
    output_path: str | Path,
    figure_title: str | None = None,
    panel_width: int = 420,
    panel_height: int = 620,
    preprocess_square_size: int | None = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_keys = [spec.key for spec in default_dataset_specs() if spec.key in selections]
    num_rows = len(ordered_keys)
    num_cols = max(len(rows) for rows in selections.values())

    left = 0.12
    right = 0.995
    bottom = 0.02
    top = 0.965 if figure_title else 0.985
    wspace = 0.03
    hspace = 0.008

    col_width_in = 2.18
    fig_width_in = col_width_in * num_cols + 0.48
    axes_width_in = fig_width_in * (right - left) / (num_cols + wspace * (num_cols - 1))
    axes_height_in = axes_width_in * (panel_height / panel_width)
    fig_height_in = axes_height_in * (num_rows + hspace * (num_rows - 1)) / (top - bottom)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width_in, fig_height_in),
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = axes[:, None]

    for row_idx, dataset_key in enumerate(ordered_keys):
        picked_rows = selections[dataset_key]
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(picked_rows):
                continue

            row = picked_rows[col_idx]
            card = _load_card(
                sample=row,
                panel_width=panel_width,
                panel_height=panel_height,
                preprocess_square_size=preprocess_square_size,
            )
            ax.imshow(card)
            if row_idx == 0:
                ax.set_title(row.category, fontsize=11, pad=4, fontweight="bold")
            ax.text(
                0.5,
                0.015,
                f"fg={row.fg_ratio * 100.0:.2f}%",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
            )

    for row_idx, dataset_key in enumerate(ordered_keys):
        label = selections[dataset_key][0].dataset_label
        first_ax = axes[row_idx, 0]
        bbox = first_ax.get_position()
        fig.text(
            bbox.x0 - 0.04,
            bbox.y0 + bbox.height / 2.0,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    if figure_title:
        fig.suptitle(figure_title, fontsize=14, y=0.99)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def dataset_overview_caption_text() -> str:
    return (
        "Representative samples from Crack500, Kaggle UAV crack dataset, and "
        "PaveCrack1300. Columns denote within-dataset crack-area quantiles "
        "(Sparse, Typical, Dense, Very dense), selected independently inside each "
        "dataset rather than at shared absolute crack ratios. Each panel shows the "
        "raw image and a crack-mask overlay; fg indicates crack foreground percentage."
    )


def dataset_overview_caption_text_360() -> str:
    return (
        "Representative samples from Crack500, Kaggle UAV crack dataset, and "
        "PaveCrack1300 after resizing each sample to 360x360, matching the standard "
        "model input used in this study. Columns still denote within-dataset crack-area "
        "quantiles (Sparse, Typical, Dense, Very dense), selected independently inside "
        "each dataset. Each panel shows the resized image and a crack-mask overlay; fg "
        "indicates crack foreground percentage measured on the native mask before resizing."
    )


B2_SUPERVISION_FROZEN_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "segformer_source_raw",
        "model": "SegFormer-B2",
        "stage": "Source-only",
        "setting": "raw",
        "labeled_samples_count": 0,
        "experiment_names": ("segformer_uav_holdout_raw_rerun", "segformer_uav_holdout_raw"),
        "metric_overrides": {
            "iou": 0.1442013288301135,
            "f1": 0.247641549696998,
            "precision": 0.1495425384196024,
            "recall": 0.7727282340564425,
        },
    },
    {
        "key": "segformer_b1_promoted",
        "model": "SegFormer-B2",
        "stage": "B1",
        "setting": "TS-bank thr080_mean082 @ 0.6",
        "labeled_samples_count": 0,
        "experiment_names": ("segformer_b2_b1_tsbank_thr080_mean082_test_thr060",),
        "metric_overrides": {
            "iou": 0.3774824520898244,
            "f1": 0.531726116225833,
            "precision": 0.5256932510270013,
            "recall": 0.5779386881798033,
        },
    },
    {
        "key": "segformer_b2_fs05",
        "model": "SegFormer-B2",
        "stage": "B2",
        "setting": "fs05",
        "labeled_samples_count": 9,
        "experiment_names": ("segformer_b2_b2_fs05_seed42_test_rerun",),
        "metric_overrides": {
            "iou": 0.5073638075873965,
            "f1": 0.6695039149314638,
            "precision": 0.6232,
            "recall": 0.7277,
        },
    },
    {
        "key": "segformer_b2_fs10",
        "model": "SegFormer-B2",
        "stage": "B2",
        "setting": "fs10",
        "labeled_samples_count": 19,
        "experiment_names": ("segformer_b2_b2_fs10_seed42_test_rerun",),
        "metric_overrides": {
            "iou": 0.5420068568653531,
            "f1": 0.6988117988147433,
            "precision": 0.6762,
            "recall": 0.7251,
        },
    },
    {
        "key": "segformer_b2_fs20",
        "model": "SegFormer-B2",
        "stage": "B2",
        "setting": "fs20",
        "labeled_samples_count": 38,
        "experiment_names": ("segformer_b2_b2_fs20_seed42_test_rerun",),
        "metric_overrides": {
            "iou": 0.5685513842673529,
            "f1": 0.7208976783449688,
            "precision": 0.6724,
            "recall": 0.7826,
        },
    },
    {
        "key": "segformer_upper_bound",
        "model": "SegFormer-B2",
        "stage": "Upper bound",
        "setting": "full-train",
        "labeled_samples_count": 189,
        "experiment_names": (
            "segformer_b2_uav_indomain_plain_360_test_rerun",
            "segformer_b2_uav_indomain_plain_360_test",
        ),
        "metric_overrides": {
            "iou": 0.5879282535068573,
            "f1": 0.7369295074826195,
            "precision": 0.6970,
            "recall": 0.7830,
        },
    },
    {
        "key": "deeplab_source_raw",
        "model": "DeepLabV3+",
        "stage": "Source-only",
        "setting": "raw",
        "labeled_samples_count": 0,
        "experiment_names": ("deeplabv3plus_plain_360_uav_holdout_raw",),
        "metric_overrides": {
            "iou": 0.12303329531162505,
            "f1": 0.2151664731994508,
            "precision": 0.13026986377579824,
            "recall": 0.6913394909056406,
        },
    },
    {
        "key": "deeplab_b1_promoted",
        "model": "DeepLabV3+",
        "stage": "B1",
        "setting": "TS-bank area1200 @ 0.5",
        "labeled_samples_count": 0,
        "experiment_names": ("deeplabv3plus_b1_tsbank_autofilter_area1200_test",),
    },
    {
        "key": "deeplab_b2_fs05",
        "model": "DeepLabV3+",
        "stage": "B2",
        "setting": "fs05_pat12",
        "labeled_samples_count": 9,
        "experiment_names": ("deeplabv3plus_b2_fs05_seed42_pat12_test",),
    },
    {
        "key": "deeplab_b2_fs10",
        "model": "DeepLabV3+",
        "stage": "B2",
        "setting": "fs10",
        "labeled_samples_count": 19,
        "experiment_names": ("deeplabv3plus_b2_fs10_seed42_test",),
    },
    {
        "key": "deeplab_b2_fs20",
        "model": "DeepLabV3+",
        "stage": "B2",
        "setting": "fs20",
        "labeled_samples_count": 38,
        "experiment_names": ("deeplabv3plus_b2_fs20_seed42_test",),
    },
    {
        "key": "deeplab_upper_bound",
        "model": "DeepLabV3+",
        "stage": "Upper bound",
        "setting": "full-train",
        "labeled_samples_count": 189,
        "experiment_names": ("deeplabv3plus_uav_indomain_plain_360_test",),
    },
)


def _load_latest_experiment_rows(csv_path: str | Path) -> dict[str, dict[str, str]]:
    latest_rows: dict[str, dict[str, str]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiment_name = row["experiment_name"]
            current = latest_rows.get(experiment_name)
            if current is None or row["timestamp_utc"] > current["timestamp_utc"]:
                latest_rows[experiment_name] = row
    return latest_rows


def _resolve_experiment_row(
    latest_rows: dict[str, dict[str, str]],
    experiment_names: tuple[str, ...],
    key: str,
) -> dict[str, str]:
    for experiment_name in experiment_names:
        row = latest_rows.get(experiment_name)
        if row is not None:
            return row
    raise KeyError(f"Missing frozen B2 supervision row for {key}: {experiment_names}")


def _parse_experiment_metric(
    row: dict[str, str],
    metric_name: str,
    metric_overrides: dict[str, float] | None,
) -> float:
    value = row.get(f"metric_{metric_name}", "")
    if value != "":
        return float(value)
    if metric_overrides is not None and metric_name in metric_overrides:
        return float(metric_overrides[metric_name])
    raise ValueError(f"Missing metric_{metric_name} for experiment {row['experiment_name']}")


def build_b2_supervision_rows(results_csv: str | Path) -> list[dict[str, object]]:
    latest_rows = _load_latest_experiment_rows(results_csv)
    rows: list[dict[str, object]] = []
    for spec in B2_SUPERVISION_FROZEN_SPECS:
        experiment_names = tuple(spec["experiment_names"])
        metric_overrides = spec.get("metric_overrides")
        row = _resolve_experiment_row(latest_rows, experiment_names, str(spec["key"]))
        rows.append(
            {
                "key": spec["key"],
                "model": spec["model"],
                "stage": spec["stage"],
                "setting": spec["setting"],
                "labeled_samples_count": int(spec["labeled_samples_count"]),
                "experiment_name": row["experiment_name"],
                "timestamp_utc": row["timestamp_utc"],
                "iou": _parse_experiment_metric(row, "iou", metric_overrides),
                "f1": _parse_experiment_metric(row, "f1", metric_overrides),
                "precision": _parse_experiment_metric(row, "precision", metric_overrides),
                "recall": _parse_experiment_metric(row, "recall", metric_overrides),
            }
        )
    return rows


def write_b2_supervision_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "model",
        "stage",
        "setting",
        "labeled_samples_count",
        "iou",
        "f1",
        "precision",
        "recall",
        "experiment_name",
        "timestamp_utc",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def render_b2_supervision_scaling(
    rows: list[dict[str, object]],
    output_path: str | Path,
    figure_title: str = "UAV hold-out supervision scaling: promoted B1/B2 recovery curves",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "SegFormer-B2": "#2f6db3",
        "DeepLabV3+": "#d97706",
    }
    x_values = [0, 9, 19, 38]
    metric_titles = [("iou", "IoU"), ("f1", "F1")]

    by_model: dict[str, dict[str, object]] = {}
    for model in colors:
        model_rows = [row for row in rows if row["model"] == model]
        by_key = {str(row["key"]): row for row in model_rows}
        prefix = "segformer" if model == "SegFormer-B2" else "deeplab"
        by_model[model] = {
            "source": by_key[f"{prefix}_source_raw"],
            "b1": by_key[f"{prefix}_b1_promoted"],
            "b2": sorted(
                [row for row in model_rows if row["stage"] == "B2"],
                key=lambda row: int(row["labeled_samples_count"]),
            ),
            "upper": by_key[f"{prefix}_upper_bound"],
        }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), sharex=True)
    for ax, (metric_key, metric_title) in zip(axes, metric_titles):
        for model, color in colors.items():
            source_row = by_model[model]["source"]
            b1_row = by_model[model]["b1"]
            b2_rows = by_model[model]["b2"]
            upper_row = by_model[model]["upper"]

            ax.scatter(
                [0],
                [float(source_row[metric_key])],
                marker="X",
                s=95,
                facecolors="white",
                edgecolors=color,
                linewidths=2,
                label=f"{model} source-only raw",
                zorder=4,
            )
            ax.plot(
                x_values,
                [float(b1_row[metric_key])] + [float(row[metric_key]) for row in b2_rows],
                marker="o",
                linewidth=2.5,
                color=color,
                label=f"{model} B1/B2",
                zorder=3,
            )
            ax.axhline(
                float(upper_row[metric_key]),
                color=color,
                linestyle="--",
                linewidth=1.9,
                alpha=0.85,
                label=f"{model} upper bound",
            )

        ax.set_title(metric_title)
        ax.set_xlabel("Labeled UAV train samples")
        ax.set_ylabel(metric_title)
        ax.set_xticks(x_values)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
