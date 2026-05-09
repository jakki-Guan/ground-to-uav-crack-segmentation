import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from dataset import CrackDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export an auto-filtered hard-negative bank from an existing mined bank "
            "using simple manifest-derived geometric rules."
        )
    )
    parser.add_argument(
        "--bank-root",
        required=True,
        help="Input bank root that contains images/, masks/, train.txt, manifest.jsonl, and summary.json.",
    )
    parser.add_argument(
        "--bank-label",
        default=None,
        help="Optional logical bank label. Defaults to the bank root directory name.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output dataset root. Defaults to generated/auto_filtered_banks/"
            "<bank_label>__<rule-suffix>."
        ),
    )
    parser.add_argument(
        "--audit-csv",
        default="results/hard_negative_audit/deeplabv3plus_tsbank_round1/audit_samples.csv",
        help=(
            "Optional audit CSV used only for overlap reporting. "
            "Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--file-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to materialize exported image and mask files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output root if it already exists.",
    )
    parser.add_argument("--min-component-area", type=float, default=0.0)
    parser.add_argument("--max-component-area", type=float, default=None)
    parser.add_argument("--min-component-mean-prob", type=float, default=0.0)
    parser.add_argument("--max-component-mean-prob", type=float, default=None)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.0)
    parser.add_argument("--max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--min-span-ratio", type=float, default=0.0)
    parser.add_argument("--max-span-ratio", type=float, default=None)
    parser.add_argument("--min-edge-distance", type=float, default=None)
    parser.add_argument("--max-edge-distance", type=float, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_numeric(entries: list[dict], field_names: list[str]) -> dict:
    if not entries:
        return {}

    summary = {}
    for field_name in field_names:
        values = [float(entry[field_name]) for entry in entries]
        summary[field_name] = {
            "mean": sum(values) / len(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def slugify_threshold(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def build_rule_suffix(args) -> str:
    parts = []
    if args.min_component_area > 0:
        parts.append(f"area{slugify_threshold(args.min_component_area)}")
    if args.max_component_area is not None:
        parts.append(f"maxarea{slugify_threshold(args.max_component_area)}")
    if args.min_component_mean_prob > 0:
        parts.append(f"mean{slugify_threshold(args.min_component_mean_prob)}")
    if args.max_component_mean_prob is not None:
        parts.append(f"maxmean{slugify_threshold(args.max_component_mean_prob)}")
    if args.min_aspect_ratio > 0:
        parts.append(f"aspect{slugify_threshold(args.min_aspect_ratio)}")
    if args.max_fill_ratio < 1.0:
        parts.append(f"fill{slugify_threshold(args.max_fill_ratio)}")
    if args.min_span_ratio > 0:
        parts.append(f"span{slugify_threshold(args.min_span_ratio)}")
    if args.max_span_ratio is not None:
        parts.append(f"maxspan{slugify_threshold(args.max_span_ratio)}")
    if args.min_edge_distance is not None:
        parts.append(f"edgege{slugify_threshold(args.min_edge_distance)}")
    if args.max_edge_distance is not None:
        parts.append(f"edgele{slugify_threshold(args.max_edge_distance)}")
    return "_".join(parts) if parts else "nofilter"


def ensure_clean_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "masks").mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, file_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if file_mode == "copy":
        shutil.copy2(src, dst)
        return
    if file_mode == "symlink":
        os.symlink(src, dst)
        return
    if file_mode == "hardlink":
        os.link(src, dst)
        return
    raise ValueError(f"Unsupported file mode: {file_mode}")


def derive_features(entry: dict, image_width: int, image_height: int) -> dict:
    bbox_x, bbox_y, bbox_w, bbox_h = [int(value) for value in entry["component_bbox_xywh"]]
    edge_distance = min(
        bbox_x,
        bbox_y,
        image_width - (bbox_x + bbox_w),
        image_height - (bbox_y + bbox_h),
    )
    bbox_width_frac = bbox_w / image_width
    bbox_height_frac = bbox_h / image_height
    span_ratio = max(bbox_width_frac, bbox_height_frac)
    return {
        "image_width": image_width,
        "image_height": image_height,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "edge_distance": edge_distance,
        "bbox_width_frac": bbox_width_frac,
        "bbox_height_frac": bbox_height_frac,
        "span_ratio": span_ratio,
        "touches_edge": int(edge_distance <= 0),
    }


def should_keep(entry: dict, args) -> bool:
    if float(entry["component_area"]) < args.min_component_area:
        return False
    if args.max_component_area is not None and float(entry["component_area"]) > args.max_component_area:
        return False
    if float(entry["component_mean_prob"]) < args.min_component_mean_prob:
        return False
    if (
        args.max_component_mean_prob is not None
        and float(entry["component_mean_prob"]) > args.max_component_mean_prob
    ):
        return False
    if float(entry["component_aspect_ratio"]) < args.min_aspect_ratio:
        return False
    if float(entry["component_fill_ratio"]) > args.max_fill_ratio:
        return False
    if float(entry["span_ratio"]) < args.min_span_ratio:
        return False
    if args.max_span_ratio is not None and float(entry["span_ratio"]) > args.max_span_ratio:
        return False
    if args.min_edge_distance is not None and float(entry["edge_distance"]) < args.min_edge_distance:
        return False
    if args.max_edge_distance is not None and float(entry["edge_distance"]) > args.max_edge_distance:
        return False
    return True


def distribution(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_audit_overlap_report(audit_rows: list[dict], selected_image_rels: set[str]) -> dict:
    selected_rows = [row for row in audit_rows if row["image_rel"] in selected_image_rels]
    removed_rows = [row for row in audit_rows if row["image_rel"] not in selected_image_rels]
    return {
        "num_audited_rows": len(audit_rows),
        "num_selected_audited_rows": len(selected_rows),
        "num_removed_audited_rows": len(removed_rows),
        "selected_layer1_distribution": distribution(
            Counter((row.get("layer1_review_label") or "").strip() or "<blank>" for row in selected_rows)
        ),
        "removed_layer1_distribution": distribution(
            Counter((row.get("layer1_review_label") or "").strip() or "<blank>" for row in removed_rows)
        ),
    }


def main():
    args = parse_args()
    bank_root = Path(args.bank_root).resolve()
    if not bank_root.exists():
        raise FileNotFoundError(f"Bank root not found: {bank_root}")

    bank_label = args.bank_label or bank_root.name
    rule_suffix = build_rule_suffix(args)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (Path("generated") / "auto_filtered_banks" / f"{bank_label}__{rule_suffix}").resolve()
    )

    manifest_path = bank_root / "manifest.jsonl"
    summary_path = bank_root / "summary.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl: {manifest_path}")

    entries = load_jsonl(manifest_path)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    source_dataset = None
    if source_summary is not None:
        dataset_root = source_summary.get("dataset_root")
        split = source_summary.get("split")
        if dataset_root and split:
            source_dataset = CrackDataset(root=dataset_root, split=split, img_size=360)

    enriched_entries = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        if source_dataset is not None and "source_index" in entry:
            raw_image, _ = source_dataset.get_raw(int(entry["source_index"]))
            image_height, image_width = raw_image.shape[:2]
        else:
            image_path = bank_root / entry["image_rel"]
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        entry.update(derive_features(entry, image_width=image_width, image_height=image_height))
        enriched_entries.append(entry)

    selected_entries = [entry for entry in enriched_entries if should_keep(entry, args)]
    if not selected_entries:
        raise ValueError("No entries survived the auto-filter. Relax the rule thresholds.")

    ensure_clean_output_dir(output_root, overwrite=args.overwrite)

    train_lines = []
    for entry in selected_entries:
        src_image = bank_root / entry["image_rel"]
        src_mask = bank_root / entry["mask_rel"]
        dst_image = output_root / entry["image_rel"]
        dst_mask = output_root / entry["mask_rel"]
        materialize_file(src_image, dst_image, args.file_mode)
        materialize_file(src_mask, dst_mask, args.file_mode)
        train_lines.append(f"{entry['image_rel']} {entry['mask_rel']}")

    train_lines.sort()
    (output_root / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for entry in sorted(selected_entries, key=lambda item: item["image_rel"]):
            export_entry = dict(entry)
            export_entry["auto_filter_bank_label"] = bank_label
            export_entry["auto_filter_rule_suffix"] = rule_suffix
            f.write(json.dumps(export_entry) + "\n")

    selected_image_rels = {entry["image_rel"] for entry in selected_entries}
    audit_overlap = None
    if args.audit_csv:
        audit_csv = Path(args.audit_csv).resolve()
        if audit_csv.exists():
            audit_rows_all = load_csv(audit_csv)
            audit_rows = [row for row in audit_rows_all if (row.get("bank_label") or "").strip() == bank_label]
            if audit_rows:
                audit_overlap = build_audit_overlap_report(audit_rows, selected_image_rels)
                with (output_root / "selected_audit_rows.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows([row for row in audit_rows if row["image_rel"] in selected_image_rels])

    summary = {
        "bank_label": bank_label,
        "source_bank_root": str(bank_root),
        "output_root": str(output_root),
        "file_mode": args.file_mode,
        "auto_filter_rules": {
            "min_component_area": args.min_component_area,
            "max_component_area": args.max_component_area,
            "min_component_mean_prob": args.min_component_mean_prob,
            "max_component_mean_prob": args.max_component_mean_prob,
            "min_aspect_ratio": args.min_aspect_ratio,
            "max_fill_ratio": args.max_fill_ratio,
            "min_span_ratio": args.min_span_ratio,
            "max_span_ratio": args.max_span_ratio,
            "min_edge_distance": args.min_edge_distance,
            "max_edge_distance": args.max_edge_distance,
        },
        "rule_suffix": rule_suffix,
        "num_source_entries": len(enriched_entries),
        "num_selected_entries": len(selected_entries),
        "selected_fraction": len(selected_entries) / len(enriched_entries),
        "selected_summary": summarize_numeric(
            selected_entries,
            [
                "component_area",
                "component_mean_prob",
                "component_fill_ratio",
                "component_aspect_ratio",
                "crop_size",
                "span_ratio",
                "edge_distance",
            ],
        ),
        "source_summary": source_summary,
        "audit_overlap": audit_overlap,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Source bank: {bank_root}")
    print(f"Bank label: {bank_label}")
    print(f"Rule suffix: {rule_suffix}")
    print(f"Output root: {output_root}")
    print(f"Selected entries: {len(selected_entries)} / {len(enriched_entries)}")
    if audit_overlap is not None:
        print(f"Audited overlap: {audit_overlap['num_selected_audited_rows']} / {audit_overlap['num_audited_rows']}")
        print(f"Selected audited layer1: {audit_overlap['selected_layer1_distribution']}")
        print(f"Removed audited layer1: {audit_overlap['removed_layer1_distribution']}")


if __name__ == "__main__":
    main()
