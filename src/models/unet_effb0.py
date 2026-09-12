"""UNet + EfficientNet-B0 baseline (ADR-007)."""
from __future__ import annotations

import torch.nn as nn

from src.models.registry import register


@register("unet_effb0")
def unet_effb0(encoder: str = "efficientnet-b0", encoder_weights: str | None = "imagenet",
               in_channels: int = 3, classes: int = 1, **_: object) -> nn.Module:
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=classes, activation=None)


@register("unet_resnet34")
def unet_resnet34(**kw: object) -> nn.Module:
    return unet_effb0(encoder="resnet34", **kw)


def set_encoder_trainable(model: nn.Module, flag: bool) -> None:
    """Freeze/unfreeze the encoder; BN goes to eval while frozen (ADR-007)."""
    for p in model.encoder.parameters():
        p.requires_grad = flag
    for m in model.encoder.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train(flag)
