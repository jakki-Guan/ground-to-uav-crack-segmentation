import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


SEGFORMER_B2_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


def canonical_model_name(model_name: str) -> str:
    normalized = (
        model_name.strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace("+", "plus")
    )
    aliases = {
        "unet": "unet",
        "fpn": "fpn",
        "linknet": "linknet",
        "pspnet": "pspnet",
        "deeplabv3plus": "deeplabv3plus",
        "segformerb2": "segformer-b2",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model name: {model_name}")
    return aliases[normalized]


def default_checkpoint_path(
    model_name: str,
    encoder_name: str = "resnet34",
    checkpoint_dir: str = "checkpoints",
) -> str:
    model_key = canonical_model_name(model_name)

    if model_key == "unet" and encoder_name == "resnet34":
        filename = "best_model.pth"
    elif model_key == "unet":
        filename = f"unet_{encoder_name}_best.pth".replace("/", "_")
    else:
        filename = f"{model_key.replace('-', '_')}_best.pth"

    return os.path.join(checkpoint_dir, filename)


class SegformerBinarySegmenter(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = SEGFORMER_B2_MODEL_NAME,
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("SegFormer-B2 currently expects 3-channel RGB input.")
        if classes != 1:
            raise ValueError("This project currently supports binary segmentation only.")

        try:
            from transformers import AutoConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required to use SegFormer-B2. "
                "Install it in the active environment first."
            ) from exc

        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                num_labels=classes,
                id2label={0: "crack"},
                label2id={"crack": 0},
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                config=config,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Failed to load SegFormer-B2 weights. "
                "Make sure the model id is correct and the weights are cached or downloadable. "
                "If you are using torch < 2.6, keep use_safetensors enabled."
            ) from exc

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


class DivisibleInputSmpSegmenter(nn.Module):
    def __init__(self, model: nn.Module, input_divisor: int):
        super().__init__()
        if input_divisor <= 0:
            raise ValueError("input_divisor must be a positive integer.")
        self.model = model
        self.input_divisor = input_divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_height, orig_width = x.shape[-2:]
        pad_height = (-orig_height) % self.input_divisor
        pad_width = (-orig_width) % self.input_divisor

        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

        logits = self.model(x)

        if pad_height or pad_width:
            logits = logits[..., :orig_height, :orig_width]

        return logits


def get_model(
    model_name: str = "Unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    pretrained_model_name: str = SEGFORMER_B2_MODEL_NAME,
):
    model_key = canonical_model_name(model_name)

    if model_key == "segformer-b2":
        return SegformerBinarySegmenter(
            pretrained_model_name=pretrained_model_name,
            in_channels=in_channels,
            classes=classes,
        )

    if model_key == "deeplabv3plus":
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
        input_divisor = getattr(base_model.encoder, "output_stride", 16)
        return DivisibleInputSmpSegmenter(
            model=base_model,
            input_divisor=input_divisor,
        )

    model_builders = {
        "unet": smp.Unet,
        "fpn": smp.FPN,
        "linknet": smp.Linknet,
        "pspnet": smp.PSPNet,
    }

    builder = model_builders[model_key]
    return builder(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
