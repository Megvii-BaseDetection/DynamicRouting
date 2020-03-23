# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from dl_lib.layers import ShapeSpec

from .backbone import (
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
)
from .meta_arch import (SemanticSegmentor, DynamicNet4Seg)
from .test_time_augmentation import DatasetMapperTTA, SemanticSegmentorWithTTA

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [
    k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")
]

assert (
    torch.Tensor([1]) == torch.Tensor([2])
).dtype == torch.bool, "Your Pytorch is too old. Please update to contain https://github.com/pytorch/pytorch/pull/21113"
