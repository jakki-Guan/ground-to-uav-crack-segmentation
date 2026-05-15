import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import matplotlib.patches as patches
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import CrackDataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYER1_LABEL_GUIDE = {
    "hard_fp": "Clearly crack-like false positive worth keeping as a hard negative.",
    "ambiguous": "Visually debatable region that could plausibly confuse a human reviewer.",
    "noise": "Texture, shadow, compression artifact, or clutter with little crack-like structure.",
    "bad_crop": "Crop framing is poor enough that the candidate is hard to judge or low-value.",
    "gt_issue": "Looks close to a labeling or boundary issue rather than a clean hard negative.",
}


LAYER2_TAXONOMY_GUIDE = {
    "pavement_edge": "Road or slab edge that forms a strong linear boundary.",
    "shadow_dark_stripe": "Shadow band or dark stripe that imitates a crack-like structure.",
    "line_like_texture": "Fine texture, scratches, or repeated line patterns that resemble cracks.",
    "surface_boundary": "Material transition, patch boundary, seam, or repair boundary.",
    "debris_object": "Debris, stain, object, or clutter region that triggers the model.",
    "other_target_clutter": "Target-domain nuisance pattern that does not fit the main buckets.",
}


AUDIT_FIELDNAMES = [
    "bank_label",
    "rank_bucket",
    "rank_overall",
    "source_sample_id",
    "score",
    "component_area",
    "component_mean_prob",
    "component_aspect_ratio",
    "component_fill_ratio",
    "crop_size",
    "crop_left",
    "crop_top",
    "component_bbox_xywh",
    "image_rel",
    "review_card_rel",
    "layer1_suggested_label",
    "layer1_review_label",
    "layer2_suggested_taxonomy",
    "layer2_review_taxonomy",
    "review_note",
]


REVIEW_FIELD_ALIASES = {
    "layer1_suggested_label": ("layer1_suggested_label", "suggested_label"),
    "layer1_review_label": ("layer1_review_label", "review_label"),
    "layer2_suggested_taxonomy": ("layer2_suggested_taxonomy",),
    "layer2_review_taxonomy": ("layer2_review_taxonomy",),
    "review_note": ("review_note",),
}


@dataclass(frozen=True)
class AuditSample:
    bank_label: str
    bank_root: Path
    dataset_root: Path
    split: str
    rank_bucket: str
    rank_overall: int
    entry: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create audit assets for mined hard-negative banks: a review CSV plus "
            "per-sample cards with crop/context/GT panels."
        )
    )
    parser.add_argument(
        "--bank",
        action="append",
        required=True,
        help=(
            "Bank directory that contains manifest.jsonl and summary.json. "
            "Repeat this flag to compare multiple banks."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional fallback dataset root when a bank summary does not define one.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional fallback split when a bank summary does not define one.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--middle-k", type=int, default=20)
    parser.add_argument("--bottom-k", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="results/hard_negative_audit/deeplabv3plus_tsbank_round1",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def slugify(raw: str) -> str:
    chars = []
    for ch in raw:
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "bank"


def derive_bank_label(bank_root: Path) -> str:
    return bank_root.name


def resolve_dataset_root(summary: dict, fallback: str | None) -> Path:
    dataset_root = summary.get("dataset_root") or fallback
    if not dataset_root:
        raise ValueError(
            "Bank summary.json does not define dataset_root and no --dataset-root fallback was provided."
        )
    return Path(dataset_root)


def resolve_split(summary: dict, fallback: str | None) -> str:
    split = summary.get("split") or fallback
    if not split:
        raise ValueError(
            "Bank summary.json does not define split and no --split fallback was provided."
        )
    return str(split)


def pick_rank_windows(num_entries: int, top_k: int, middle_k: int, bottom_k: int) -> list[tuple[str, list[int]]]:
    total_requested = top_k + middle_k + bottom_k
    if num_entries <= total_requested:
        top_end = int(np.ceil(num_entries / 3.0))
        middle_end = int(np.ceil(2.0 * num_entries / 3.0))
        strata = {
            "top": list(range(0, top_end)),
            "middle": list(range(top_end, middle_end)),
            "bottom": list(range(middle_end, num_entries)),
        }
        limits = {
            "top": top_k,
            "middle": middle_k,
            "bottom": bottom_k,
        }
        sampled = []
        for rank_bucket in ("top", "middle", "bottom"):
            indices = strata[rank_bucket]
            limit = limits[rank_bucket]
            if len(indices) <= limit:
                selected = indices
            elif rank_bucket == "top":
                selected = indices[:limit]
            elif rank_bucket == "middle":
                center = len(indices) // 2
                start = max(0, center - limit // 2)
                end = start + limit
                if end > len(indices):
                    end = len(indices)
                    start = end - limit
                selected = indices[start:end]
            else:
                selected = indices[-limit:]
            sampled.append((rank_bucket, selected))
        return sampled

    top_end = min(top_k, num_entries)
    top_indices = list(range(top_end))

    bottom_start = max(num_entries - bottom_k, 0)
    while bottom_start < num_entries and bottom_start in top_indices:
        bottom_start += 1
    bottom_indices = list(range(bottom_start, num_entries))

    reserved = set(top_indices) | set(bottom_indices)
    remaining = [idx for idx in range(num_entries) if idx not in reserved]

    if not remaining:
        middle_indices = []
    elif len(remaining) <= middle_k:
        middle_indices = remaining
    else:
        center = len(remaining) // 2
        start = max(0, center - middle_k // 2)
        end = start + middle_k
        if end > len(remaining):
            end = len(remaining)
            start = end - middle_k
        middle_indices = remaining[start:end]

    return [
        ("top", top_indices),
        ("middle", middle_indices),
        ("bottom", bottom_indices),
    ]


def build_samples_for_bank(
    bank_root: Path,
    dataset_root_fallback: str | None,
    split_fallback: str | None,
    top_k: int,
    middle_k: int,
    bottom_k: int,
) -> tuple[list[AuditSample], dict]:
    summary_path = bank_root / "summary.json"
    manifest_path = bank_root / "manifest.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl: {manifest_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = load_jsonl(manifest_path)
    entries.sort(key=lambda row: row["score"], reverse=True)

    dataset_root = resolve_dataset_root(summary, dataset_root_fallback)
    split = resolve_split(summary, split_fallback)
    bank_label = derive_bank_label(bank_root)

    samples = []
    for rank_bucket, indices in pick_rank_windows(len(entries), top_k, middle_k, bottom_k):
        for idx in indices:
            samples.append(
                AuditSample(
                    bank_label=bank_label,
                    bank_root=bank_root,
                    dataset_root=dataset_root,
                    split=split,
                    rank_bucket=rank_bucket,
                    rank_overall=idx + 1,
                    entry=entries[idx],
                )
            )

    summary_row = {
        "bank_label": bank_label,
        "bank_root": str(bank_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "num_entries": len(entries),
        "num_selected_for_audit": len(samples),
        "component_area_mean": summary.get("selected_summary", {}).get("component_area_mean", ""),
        "component_area_median": summary.get("selected_summary", {}).get("component_area_median", ""),
        "component_mean_prob_mean": summary.get("selected_summary", {}).get("component_mean_prob_mean", ""),
        "component_mean_prob_median": summary.get("selected_summary", {}).get("component_mean_prob_median", ""),
        "crop_size_mean": summary.get("selected_summary", {}).get("crop_size_mean", ""),
        "crop_size_median": summary.get("selected_summary", {}).get("crop_size_median", ""),
        "candidate_count_before_global_cap": summary.get("candidate_count_before_global_cap", ""),
        "component_threshold": summary.get("component_threshold", ""),
        "min_component_mean_prob": summary.get("min_component_mean_prob", ""),
        "crop_context_scale": summary.get("crop_context_scale", ""),
    }
    return samples, summary_row


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        overlay_color = np.array(color, dtype=np.float32)
        output[mask_bool] = (1.0 - alpha) * output[mask_bool] + alpha * overlay_color
    return np.clip(output, 0, 255).astype(np.uint8)


def load_crop_image(bank_root: Path, image_rel: str) -> np.ndarray:
    image_path = bank_root / image_rel
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def draw_review_card(sample: AuditSample, dataset: CrackDataset, output_path: Path):
    entry = sample.entry
    source_index = int(entry["source_index"])
    raw_image, raw_mask = dataset.get_raw(source_index)
    crop_image = load_crop_image(sample.bank_root, entry["image_rel"])

    crop_left = int(entry["crop_left"])
    crop_top = int(entry["crop_top"])
    crop_size = int(entry["crop_size"])
    bbox_x, bbox_y, bbox_w, bbox_h = [int(v) for v in entry["component_bbox_xywh"]]

    context_with_gt = overlay_mask(raw_image, raw_mask, color=(0, 255, 0), alpha=0.35)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(crop_image)
    axes[0, 0].set_title("Crop")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(raw_image)
    axes[0, 1].add_patch(
        patches.Rectangle(
            (crop_left, crop_top),
            crop_size,
            crop_size,
            linewidth=2.0,
            edgecolor="yellow",
            facecolor="none",
        )
    )
    axes[0, 1].add_patch(
        patches.Rectangle(
            (bbox_x, bbox_y),
            bbox_w,
            bbox_h,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
    )
    axes[0, 1].set_title("Source Context")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(context_with_gt)
    axes[1, 0].add_patch(
        patches.Rectangle(
            (crop_left, crop_top),
            crop_size,
            crop_size,
            linewidth=2.0,
            edgecolor="yellow",
            facecolor="none",
        )
    )
    axes[1, 0].add_patch(
        patches.Rectangle(
            (bbox_x, bbox_y),
            bbox_w,
            bbox_h,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
    )
    axes[1, 0].set_title("Source + GT Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].axis("off")
    metadata_lines = [
        f"bank: {sample.bank_label}",
        f"bucket: {sample.rank_bucket}",
        f"rank_overall: {sample.rank_overall}",
        f"source_sample_id: {entry['source_sample_id']}",
        f"score: {entry['score']:.4f}",
        f"component_area: {entry['component_area']}",
        f"component_mean_prob: {entry['component_mean_prob']:.4f}",
        f"component_aspect_ratio: {entry['component_aspect_ratio']:.4f}",
        f"component_fill_ratio: {entry['component_fill_ratio']:.4f}",
        f"crop_xy_size: ({crop_left}, {crop_top}, {crop_size})",
        f"component_bbox_xywh: {entry['component_bbox_xywh']}",
        f"crop_gt_foreground_pixels: {entry['crop_gt_foreground_pixels']}",
    ]
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(metadata_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(f"{sample.bank_label} | {sample.rank_bucket} | {entry['image_rel']}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_bank_overview_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "bank_label",
        "bank_root",
        "dataset_root",
        "split",
        "num_entries",
        "num_selected_for_audit",
        "component_area_mean",
        "component_area_median",
        "component_mean_prob_mean",
        "component_mean_prob_median",
        "crop_size_mean",
        "crop_size_median",
        "candidate_count_before_global_cap",
        "component_threshold",
        "min_component_mean_prob",
        "crop_context_scale",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def audit_row_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("bank_label", "")),
        str(row.get("source_sample_id", "")),
        str(row.get("image_rel", "")),
    )


def load_existing_review_annotations(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        return {}

    annotations = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = audit_row_key(row)
            if not any(key):
                continue
            annotations[key] = {
                field: next((row.get(alias, "") for alias in aliases if row.get(alias, "")), "")
                for field, aliases in REVIEW_FIELD_ALIASES.items()
            }
    return annotations


def merge_existing_review_annotations(
    rows: list[dict], existing_annotations: dict[tuple[str, str, str], dict[str, str]]
) -> list[dict]:
    for row in rows:
        existing = existing_annotations.get(audit_row_key(row))
        if not existing:
            continue
        for field in REVIEW_FIELD_ALIASES:
            row[field] = existing.get(field, row.get(field, ""))
    return rows


def write_readme(path: Path):
    lines = [
        "# Hard-Negative Audit Package",
        "",
        "## Goal",
        "",
        "Check whether mined crops are genuinely crack-like high-confidence false positives or mostly noise,",
        "and then assign interpretable nuisance categories for paper-facing failure diagnosis.",
        "",
        "## Layer 1 Labels",
        "",
    ]
    for label, description in LAYER1_LABEL_GUIDE.items():
        lines.append(f"- `{label}`: {description}")
    lines.extend(
        [
            "",
            "## Layer 2 Nuisance Taxonomy",
            "",
        ]
    )
    for label, description in LAYER2_TAXONOMY_GUIDE.items():
        lines.append(f"- `{label}`: {description}")
    lines.extend(
        [
            "",
            "## Suggested Workflow",
            "",
            "1. Open `bank_overview.csv` to compare broad bank statistics.",
            "2. Review `review_cards/` in top, middle, and bottom buckets for each bank.",
            "3. Fill `layer1_review_label` for every sampled crop in `audit_samples.csv`.",
            "4. Fill `layer2_review_taxonomy` whenever the visual nuisance pattern is identifiable.",
            "5. Leave layer 2 blank for crops that are mainly `bad_crop` or `gt_issue`.",
            "6. Use `review_note` only for representative edge cases or uncertain examples.",
            "",
            "## Decision Rule",
            "",
            "- Promote a bank only if layer 1 stays visually dominated by `hard_fp` / `ambiguous` examples.",
            "- Pause further hold-out testing if `noise` + `bad_crop` becomes a large share of the selected crops.",
            "- Use layer 2 to report what kinds of target-domain nuisance structures dominate the false positives.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    review_dir = output_dir / "review_cards"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_annotations = load_existing_review_annotations(output_dir / "audit_samples.csv")

    all_samples = []
    overview_rows = []
    for raw_bank in args.bank:
        bank_root = Path(raw_bank)
        samples, summary_row = build_samples_for_bank(
            bank_root=bank_root,
            dataset_root_fallback=args.dataset_root,
            split_fallback=args.split,
            top_k=args.top_k,
            middle_k=args.middle_k,
            bottom_k=args.bottom_k,
        )
        all_samples.extend(samples)
        overview_rows.append(summary_row)

    datasets: dict[tuple[str, str], CrackDataset] = {}
    audit_rows = []

    for sample in all_samples:
        dataset_key = (str(sample.dataset_root.resolve()), sample.split)
        if dataset_key not in datasets:
            datasets[dataset_key] = CrackDataset(
                root=str(sample.dataset_root),
                split=sample.split,
                img_size=360,
            )
        dataset = datasets[dataset_key]

        bank_slug = slugify(sample.bank_label)
        image_stem = Path(sample.entry["image_rel"]).stem
        card_rel = Path("review_cards") / bank_slug / sample.rank_bucket / f"{image_stem}.png"
        card_path = output_dir / card_rel
        draw_review_card(sample, dataset, card_path)

        audit_rows.append(
            {
                "bank_label": sample.bank_label,
                "rank_bucket": sample.rank_bucket,
                "rank_overall": sample.rank_overall,
                "source_sample_id": sample.entry["source_sample_id"],
                "score": f"{sample.entry['score']:.6f}",
                "component_area": sample.entry["component_area"],
                "component_mean_prob": f"{sample.entry['component_mean_prob']:.6f}",
                "component_aspect_ratio": f"{sample.entry['component_aspect_ratio']:.6f}",
                "component_fill_ratio": f"{sample.entry['component_fill_ratio']:.6f}",
                "crop_size": sample.entry["crop_size"],
                "crop_left": sample.entry["crop_left"],
                "crop_top": sample.entry["crop_top"],
                "component_bbox_xywh": json.dumps(sample.entry["component_bbox_xywh"]),
                "image_rel": sample.entry["image_rel"],
                "review_card_rel": str(card_rel),
                "layer1_suggested_label": "",
                "layer1_review_label": "",
                "layer2_suggested_taxonomy": "",
                "layer2_review_taxonomy": "",
                "review_note": "",
            }
        )

    merge_existing_review_annotations(audit_rows, existing_annotations)
    write_bank_overview_csv(output_dir / "bank_overview.csv", overview_rows)
    write_audit_csv(output_dir / "audit_samples.csv", audit_rows)
    write_readme(output_dir / "README.md")

    print(f"Saved audit package to: {output_dir}")
    print(f"Review cards directory: {review_dir}")
    print(f"Audit CSV: {output_dir / 'audit_samples.csv'}")


if __name__ == "__main__":
    main()
