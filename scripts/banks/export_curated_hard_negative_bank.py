import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


VALID_LAYER1_LABELS = {
    "hard_fp",
    "ambiguous",
    "noise",
    "bad_crop",
    "gt_issue",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export curated hard-negative banks from a completed audit package. "
            "The script filters audit_samples.csv by reviewed layer-1 labels and "
            "writes dataset roots that can be consumed directly by train.py."
        )
    )
    parser.add_argument(
        "--audit-csv",
        default="results/hard_negative_audit/deeplabv3plus_tsbank_round1/audit_samples.csv",
        help="Audit CSV produced by scripts/banks/make_hard_negative_audit_assets.py.",
    )
    parser.add_argument(
        "--bank-overview-csv",
        default=None,
        help=(
            "Optional bank_overview.csv. Defaults to the sibling file next to --audit-csv. "
            "Used to resolve each bank_label back to its source bank root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory that will contain one curated dataset root per bank. "
            "Defaults to <audit package>/curated_banks."
        ),
    )
    parser.add_argument(
        "--bank-label",
        action="append",
        default=None,
        help=(
            "Restrict export to one or more bank_label values. "
            "Repeat this flag to export multiple banks."
        ),
    )
    parser.add_argument(
        "--keep-layer1",
        nargs="+",
        default=["hard_fp", "ambiguous"],
        help=(
            "Reviewed layer-1 labels to keep. "
            "Example: --keep-layer1 hard_fp ambiguous"
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
        help="Overwrite an existing curated output root if it already exists.",
    )
    return parser.parse_args()


def validate_keep_labels(keep_labels: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for label in keep_labels:
        normalized = label.strip()
        if not normalized:
            continue
        if normalized not in VALID_LAYER1_LABELS:
            raise ValueError(
                f"Unsupported layer-1 label: {normalized}. "
                f"Choose from: {sorted(VALID_LAYER1_LABELS)}"
            )
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)

    if not cleaned:
        raise ValueError("--keep-layer1 must include at least one valid label.")
    return cleaned


def resolve_default_paths(args):
    audit_csv = Path(args.audit_csv).resolve()
    bank_overview_csv = (
        Path(args.bank_overview_csv).resolve()
        if args.bank_overview_csv
        else audit_csv.parent / "bank_overview.csv"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else audit_csv.parent / "curated_banks"
    )
    return audit_csv, bank_overview_csv, output_dir


def load_audit_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Audit CSV is empty: {path}")
    return rows


def load_bank_roots(path: Path) -> dict[str, Path]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"bank_overview.csv is empty: {path}")

    mapping = {}
    for row in rows:
        bank_label = (row.get("bank_label") or "").strip()
        bank_root = (row.get("bank_root") or "").strip()
        if not bank_label or not bank_root:
            raise ValueError(f"Missing bank_label/bank_root in {path}: {row}")
        mapping[bank_label] = Path(bank_root).resolve()
    return mapping


def load_manifest_entries(bank_root: Path) -> dict[str, dict]:
    manifest_path = bank_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.jsonl: {manifest_path}")

    manifest_by_image = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            image_rel = entry.get("image_rel")
            if not image_rel:
                raise ValueError(f"Manifest entry missing image_rel in {manifest_path}: {entry}")
            manifest_by_image[image_rel] = entry
    return manifest_by_image


def summarize_selected_entries(entries: list[dict]) -> dict:
    if not entries:
        return {
            "num_crops": 0,
            "unique_source_images": 0,
        }

    component_areas = [entry["component_area"] for entry in entries]
    mean_probs = [entry["component_mean_prob"] for entry in entries]
    crop_sizes = [entry["crop_size"] for entry in entries]
    return {
        "num_crops": len(entries),
        "unique_source_images": len({entry["source_sample_id"] for entry in entries}),
        "component_area_mean": sum(component_areas) / len(component_areas),
        "component_area_median": median(component_areas),
        "component_mean_prob_mean": sum(mean_probs) / len(mean_probs),
        "component_mean_prob_median": median(mean_probs),
        "crop_size_mean": sum(crop_sizes) / len(crop_sizes),
        "crop_size_median": median(crop_sizes),
    }


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def output_suffix_from_labels(keep_labels: list[str]) -> str:
    return "_".join(keep_labels)


def ensure_clean_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Pass --overwrite to replace it."
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


def normalize_int(value: str) -> int:
    return int(value.strip())


def distribution(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def export_bank(
    audit_csv: Path,
    bank_label: str,
    bank_root: Path,
    bank_rows: list[dict],
    keep_labels: list[str],
    output_dir: Path,
    file_mode: str,
    overwrite: bool,
):
    selected_rows = [
        row
        for row in bank_rows
        if (row.get("layer1_review_label") or "").strip() in keep_labels
    ]
    if not selected_rows:
        raise ValueError(
            f"No rows kept for bank={bank_label} with keep labels {keep_labels}."
        )

    manifest_by_image = load_manifest_entries(bank_root)
    suffix = output_suffix_from_labels(keep_labels)
    export_root = output_dir / f"{bank_label}__{suffix}"
    ensure_clean_output_dir(export_root, overwrite=overwrite)

    curated_manifest_entries = []
    train_lines = []
    layer1_counts = Counter()
    layer2_counts = Counter()

    for row in selected_rows:
        image_rel = (row.get("image_rel") or "").strip()
        if not image_rel:
            raise ValueError(f"Missing image_rel in audit row: {row}")
        if image_rel not in manifest_by_image:
            raise KeyError(
                f"image_rel={image_rel} from audit CSV not found in {bank_root / 'manifest.jsonl'}"
            )

        manifest_entry = dict(manifest_by_image[image_rel])
        mask_rel = manifest_entry.get("mask_rel")
        if not mask_rel:
            raise ValueError(
                f"Manifest entry missing mask_rel for bank={bank_label}, image_rel={image_rel}"
            )

        src_image = bank_root / image_rel
        src_mask = bank_root / mask_rel
        dst_image = export_root / image_rel
        dst_mask = export_root / mask_rel
        if not src_image.exists():
            raise FileNotFoundError(f"Missing source image: {src_image}")
        if not src_mask.exists():
            raise FileNotFoundError(f"Missing source mask: {src_mask}")

        materialize_file(src_image, dst_image, file_mode=file_mode)
        materialize_file(src_mask, dst_mask, file_mode=file_mode)
        train_lines.append(f"{image_rel} {mask_rel}")

        layer1_label = (row.get("layer1_review_label") or "").strip()
        layer2_label = (row.get("layer2_review_taxonomy") or "").strip()
        layer1_counts[layer1_label] += 1
        if layer2_label:
            layer2_counts[layer2_label] += 1

        manifest_entry.update(
            {
                "bank_label": bank_label,
                "source_bank_root": str(bank_root),
                "audit_rank_bucket": (row.get("rank_bucket") or "").strip(),
                "audit_rank_overall": normalize_int(row["rank_overall"]),
                "audit_layer1_suggested_label": (row.get("layer1_suggested_label") or "").strip(),
                "audit_layer1_review_label": layer1_label,
                "audit_layer2_suggested_taxonomy": (
                    row.get("layer2_suggested_taxonomy") or ""
                ).strip(),
                "audit_layer2_review_taxonomy": layer2_label,
                "audit_review_note": (row.get("review_note") or "").strip(),
                "audit_review_card_rel": (row.get("review_card_rel") or "").strip(),
            }
        )
        curated_manifest_entries.append(manifest_entry)

    curated_manifest_entries.sort(key=lambda entry: entry["image_rel"])
    train_lines.sort()

    (export_root / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    with (export_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for entry in curated_manifest_entries:
            f.write(json.dumps(entry) + "\n")

    selected_csv_fieldnames = list(selected_rows[0].keys())
    with (export_root / "selected_audit_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_csv_fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    source_summary_path = bank_root / "summary.json"
    source_summary = None
    if source_summary_path.exists():
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))

    summary = {
        "audit_csv": str(audit_csv),
        "bank_label": bank_label,
        "source_bank_root": str(bank_root),
        "keep_layer1": keep_labels,
        "file_mode": file_mode,
        "num_audit_rows_for_bank": len(bank_rows),
        "num_selected_rows": len(selected_rows),
        "selected_fraction_of_audited_rows": len(selected_rows) / len(bank_rows),
        "layer1_distribution": distribution(layer1_counts),
        "layer2_distribution": distribution(layer2_counts),
        "selected_summary": summarize_selected_entries(curated_manifest_entries),
        "source_summary": source_summary,
    }
    (export_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return {
        "bank_label": bank_label,
        "export_root": export_root,
        "num_selected_rows": len(selected_rows),
        "num_audit_rows_for_bank": len(bank_rows),
        "layer1_distribution": distribution(layer1_counts),
    }


def main():
    args = parse_args()
    keep_labels = validate_keep_labels(args.keep_layer1)
    audit_csv, bank_overview_csv, output_dir = resolve_default_paths(args)

    audit_rows = load_audit_rows(audit_csv)
    bank_roots = load_bank_roots(bank_overview_csv)

    requested_bank_labels = args.bank_label or sorted({row["bank_label"] for row in audit_rows})
    missing = [label for label in requested_bank_labels if label not in bank_roots]
    if missing:
        raise KeyError(
            f"bank_label values not found in {bank_overview_csv}: {missing}"
        )

    rows_by_bank = {}
    for bank_label in requested_bank_labels:
        rows_by_bank[bank_label] = [
            row for row in audit_rows if (row.get("bank_label") or "").strip() == bank_label
        ]
        if not rows_by_bank[bank_label]:
            raise ValueError(f"No audit rows found for bank_label={bank_label}")

    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for bank_label in requested_bank_labels:
        reports.append(
            export_bank(
                audit_csv=audit_csv,
                bank_label=bank_label,
                bank_root=bank_roots[bank_label],
                bank_rows=rows_by_bank[bank_label],
                keep_labels=keep_labels,
                output_dir=output_dir,
                file_mode=args.file_mode,
                overwrite=args.overwrite,
            )
        )

    print(f"Audit CSV: {audit_csv}")
    print(f"Bank overview: {bank_overview_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Keep labels: {keep_labels}")
    for report in reports:
        print(
            f"- {report['bank_label']}: "
            f"{report['num_selected_rows']} / {report['num_audit_rows_for_bank']} selected -> "
            f"{report['export_root']}"
        )
        print(f"  layer1: {report['layer1_distribution']}")


if __name__ == "__main__":
    main()
