import argparse
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Hugging Face nvidia/mit-b5 ImageNet weights to the "
            "DAFormer/MMSeg MixVisionTransformer key format."
        )
    )
    parser.add_argument("--repo-id", default="nvidia/mit-b5")
    parser.add_argument("--filename", default="pytorch_model.bin")
    parser.add_argument("--local-file", default=None)
    parser.add_argument("--output", default="external/DAFormer/pretrained/mit_b5.pth")
    return parser.parse_args()


def map_patch_embedding(key):
    match = re.match(r"segformer\.encoder\.patch_embeddings\.(\d+)\.(.+)", key)
    if not match:
        return None
    stage = int(match.group(1)) + 1
    suffix = match.group(2).replace("layer_norm", "norm")
    return f"patch_embed{stage}.{suffix}"


def map_block_key(key, state_dict):
    match = re.match(r"segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)", key)
    if not match:
        return None, None

    stage = int(match.group(1)) + 1
    layer = match.group(2)
    suffix = match.group(3)
    prefix = f"block{stage}.{layer}."

    replacements = [
        ("layer_norm_1.", "norm1."),
        ("layer_norm_2.", "norm2."),
        ("attention.self.query.", "attn.q."),
        ("attention.self.sr.", "attn.sr."),
        ("attention.self.layer_norm.", "attn.norm."),
        ("attention.output.dense.", "attn.proj."),
        ("mlp.dense1.", "mlp.fc1."),
        ("mlp.dense2.", "mlp.fc2."),
        ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv."),
    ]

    if suffix.startswith("attention.self.key."):
        value_key = key.replace("attention.self.key.", "attention.self.value.")
        if value_key not in state_dict:
            raise KeyError(f"Missing paired value tensor for {key}")
        param_name = suffix.replace("attention.self.key.", "")
        return prefix + "attn.kv." + param_name, torch.cat([state_dict[key], state_dict[value_key]], dim=0)

    if suffix.startswith("attention.self.value."):
        return None, None

    for old, new in replacements:
        if suffix.startswith(old):
            return prefix + suffix.replace(old, new, 1), state_dict[key]

    return None, None


def map_final_norm(key):
    match = re.match(r"segformer\.encoder\.layer_norm\.(\d+)\.(.+)", key)
    if not match:
        return None
    stage = int(match.group(1)) + 1
    return f"norm{stage}.{match.group(2)}"


def convert_state_dict(hf_state_dict):
    converted = {}
    skipped = []
    for key in hf_state_dict.keys():
        mapped_key = map_patch_embedding(key)
        if mapped_key is not None:
            converted[mapped_key] = hf_state_dict[key]
            continue

        mapped_key, mapped_value = map_block_key(key, hf_state_dict)
        if mapped_key is not None:
            converted[mapped_key] = mapped_value
            continue
        if mapped_value is None and ".attention.self.value." in key:
            continue

        mapped_key = map_final_norm(key)
        if mapped_key is not None:
            converted[mapped_key] = hf_state_dict[key]
            continue

        skipped.append(key)

    return converted, skipped


def main():
    args = parse_args()
    if args.local_file:
        input_path = args.local_file
    else:
        input_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    hf_state_dict = torch.load(input_path, map_location="cpu")
    converted, skipped = convert_state_dict(hf_state_dict)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Converted tensors: {len(converted)}")
    if skipped:
        print("Skipped tensors:")
        for key in skipped:
            print(f"  {key}")


if __name__ == "__main__":
    main()
