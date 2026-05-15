import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


LAYER1_ORDER = [
    "hard_fp",
    "ambiguous",
    "noise",
    "bad_crop",
    "gt_issue",
]

LAYER2_ORDER = [
    "pavement_edge",
    "shadow_dark_stripe",
    "line_like_texture",
    "surface_boundary",
    "debris_object",
    "other_target_clutter",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a completed hard-negative audit CSV so reviewed bank quality "
            "and nuisance taxonomy can be compared before follow-up training."
        )
    )
    parser.add_argument(
        "--audit-csv",
        required=True,
        help="Path to audit_samples.csv produced by scripts/banks/make_hard_negative_audit_assets.py.",
    )
    parser.add_argument(
        "--bank-label",
        action="append",
        default=None,
        help=(
            "Optional bank_label to include. Repeat this flag to summarize multiple banks. "
            "Defaults to all banks in the CSV."
        ),
    )
    parser.add_argument(
        "--keep-layer1",
        nargs="+",
        default=["hard_fp", "ambiguous"],
        help=(
            "Layer-1 labels considered keepable for follow-up training summaries. "
            "Default: hard_fp ambiguous"
        ),
    )
    parser.add_argument(
        "--top-k-taxonomy",
        type=int,
        default=6,
        help="How many nuisance taxonomy categories to print per summary block.",
    )
    return parser.parse_args()


def ordered_items(counter: Counter, preferred_order: list[str]) -> list[tuple[str, int]]:
    seen = set()
    items = []
    for key in preferred_order:
        if counter.get(key, 0):
            items.append((key, counter[key]))
            seen.add(key)
    for key in sorted(counter):
        if key not in seen and counter[key]:
            items.append((key, counter[key]))
    return items


def format_counter(counter: Counter, preferred_order: list[str], limit: int | None = None) -> str:
    items = ordered_items(counter, preferred_order)
    if limit is not None:
        items = items[:limit]
    if not items:
        return "<empty>"
    return ", ".join(f"{key}={value}" for key, value in items)


def main():
    args = parse_args()
    audit_csv = Path(args.audit_csv).resolve()
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")

    with audit_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Audit CSV is empty: {audit_csv}")

    wanted_banks = set(args.bank_label) if args.bank_label else None
    keep_layer1 = {label.strip() for label in args.keep_layer1 if label.strip()}

    grouped = defaultdict(list)
    for row in rows:
        bank_label = (row.get("bank_label") or "").strip()
        if not bank_label:
            raise ValueError("Encountered audit row without bank_label.")
        if wanted_banks is not None and bank_label not in wanted_banks:
            continue
        grouped[bank_label].append(row)

    if not grouped:
        raise ValueError("No matching banks found in audit CSV for the requested filters.")

    print(f"Audit CSV: {audit_csv}")
    print(f"Selected banks: {len(grouped)}")

    for bank_label in sorted(grouped):
        bank_rows = grouped[bank_label]
        layer1_counts = Counter((row.get("layer1_review_label") or "").strip() for row in bank_rows)
        layer2_all = Counter((row.get("layer2_review_taxonomy") or "").strip() for row in bank_rows)
        keep_rows = [
            row for row in bank_rows if (row.get("layer1_review_label") or "").strip() in keep_layer1
        ]
        layer2_keep = Counter((row.get("layer2_review_taxonomy") or "").strip() for row in keep_rows)

        total = len(bank_rows)
        keep_count = len(keep_rows)
        keep_share = 100.0 * keep_count / total if total else 0.0

        print()
        print(bank_label)
        print(f"  total: {total}")
        print(f"  layer1: {format_counter(layer1_counts, LAYER1_ORDER)}")
        print(f"  keepable({'+'.join(sorted(keep_layer1))}): {keep_count}/{total} = {keep_share:.1f}%")
        print(
            f"  layer2 all: {format_counter(layer2_all, LAYER2_ORDER, limit=args.top_k_taxonomy)}"
        )
        print(
            f"  layer2 keepable: {format_counter(layer2_keep, LAYER2_ORDER, limit=args.top_k_taxonomy)}"
        )


if __name__ == "__main__":
    main()
