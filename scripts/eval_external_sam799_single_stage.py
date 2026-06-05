import argparse
import json
import shutil
import sys
from pathlib import Path

from eval_external_sam799_patchwise import (
    StageSpec,
    build_eval_config,
    build_patch_plan_rows,
    evaluate_stage,
    load_samples,
    resolve_device,
    write_csv,
    write_profiling_summary,
    write_readme,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-protocol patch-based external evaluation on SAM799-CVAT "
            "for a single repo-native checkpoint."
        )
    )
    parser.add_argument("--dataset-root", default="SAM799_CVAT")
    parser.add_argument("--split", default="external_test")
    parser.add_argument(
        "--manifest-csv",
        default="results/external_sam799_cvat_patchwise/sam799_manifest.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-key", required=True)
    parser.add_argument("--stage-label", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--model-name", default="deeplabv3plus")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--model-input-size", type=int, default=360)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-probability-maps", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--save-limit", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def normalize_encoder_weights(raw: str | None):
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return raw


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_spec = StageSpec(
        key=args.stage_key,
        label=args.stage_label,
        checkpoint_path=args.checkpoint_path,
        threshold=args.threshold,
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=normalize_encoder_weights(args.encoder_weights),
    )

    samples = load_samples(
        dataset_root=dataset_root,
        split_name=args.split,
        max_images=args.max_images,
    )
    patch_plans, patch_plan_rows = build_patch_plan_rows(
        samples=samples,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Split: {args.split}")
    print(f"Images: {len(samples)}")
    print(f"Stage: {stage_spec.key}")
    print(f"Patch size / stride: {args.patch_size} / {args.stride}")

    eval_config = build_eval_config(args=args, samples=samples, stage_specs=[stage_spec])
    (output_dir / "eval_config.json").write_text(json.dumps(eval_config, indent=2), encoding="utf-8")
    write_readme(output_dir=output_dir, args=args, stage_specs=[stage_spec], samples=samples)
    write_csv(output_dir / "per_image_patch_counts.csv", patch_plan_rows)

    aggregate_row, per_image_rows = evaluate_stage(
        stage_spec=stage_spec,
        samples=samples,
        patch_plans=patch_plans,
        args=args,
        output_dir=output_dir,
        device=device,
    )
    write_csv(output_dir / "aggregate_metrics.csv", [aggregate_row])
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_profiling_summary(output_dir=output_dir, aggregate_rows=[aggregate_row])

    manifest_path = Path(args.manifest_csv).resolve()
    if manifest_path.exists():
        manifest_copy_path = output_dir / "sam799_manifest.csv"
        if manifest_copy_path.resolve() != manifest_path:
            shutil.copy2(manifest_path, manifest_copy_path)
        print(f"Manifest already present at: {manifest_path}")
    else:
        print(
            "Manifest CSV not found in the requested location. "
            "Run scripts/data/build_sam799_manifest.py to create it."
        )

    print(f"Wrote eval config to: {output_dir / 'eval_config.json'}")
    print(f"Wrote patch-count rows to: {output_dir / 'per_image_patch_counts.csv'}")
    print(f"Wrote per-image metrics to: {output_dir / 'per_image_metrics.csv'}")
    print(f"Wrote aggregate metrics to: {output_dir / 'aggregate_metrics.csv'}")
    print(f"Wrote profiling summary to: {output_dir / 'profiling_summary.md'}")


if __name__ == "__main__":
    main()
