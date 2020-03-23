# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List

import torch
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .lr_scheduler import PolyLR, WarmupCosineLR, WarmupMultiStepLR


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.SOLVER.OPTIMIZER
    """
    if cfg.NAME == "SGD":
        params: List[Dict[str, Any]] = []
        for key, value in model.named_parameters():
            if not cfg.get("WEIGHT_DECAY_CONV_ONLY", False):
                if not value.requires_grad:
                    continue
                lr = cfg.BASE_LR
                weight_decay = cfg.WEIGHT_DECAY
                if key.endswith("norm.weight") or key.endswith("norm.bias"):
                    weight_decay = cfg.WEIGHT_DECAY_NORM
                elif key.endswith(".bias"):
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.BASE_LR * cfg.BIAS_LR_FACTOR
                    weight_decay = cfg.WEIGHT_DECAY_BIAS
            else:
                lr = cfg.BASE_LR
                if "conv.weight" not in key:
                    weight_decay = 0
                else:
                    weight_decay = cfg.WEIGHT_DECAY
            # multiply lr for gating function
            if "GATE_LR_MULTI" in cfg:
                if cfg.GATE_LR_MULTI > 0.0 and "gate_conv" in key:
                    lr *= cfg.GATE_LR_MULTI

            params += [{
                "params": [value],
                "lr": lr,
                "weight_decay": weight_decay
            }]
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.MOMENTUM)
    elif cfg.NAME == "AdamW":
        lr = cfg.BASE_LR
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     betas=cfg.BETAS,
                                     weight_decay=cfg.WEIGHT_DECAY,
                                     amsgrad=cfg.AMSGRAD)
    return optimizer


def build_lr_scheduler(
    cfg, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.STEPS,
            cfg.GAMMA,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.MAX_ITER,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD,
        )
    elif name == "LambdaLR":
        return LambdaLR(optimizer, cfg.LAMBDA_SCHEDULE)
    elif name == "OneCycleLR":
        return OneCycleLR(optimizer,
                          cfg.MAX_LR,
                          total_steps=cfg.MAX_ITER,
                          pct_start=cfg.PCT_START,
                          base_momentum=cfg.BASE_MOM,
                          max_momentum=cfg.MAX_MOM,
                          div_factor=cfg.DIV_FACTOR)
    elif name == "PolyLR":
        return PolyLR(
            optimizer,
            cfg.MAX_ITER,
            cfg.POLY_POWER,
            warmup_factor=cfg.WARMUP_FACTOR,
            warmup_iters=cfg.WARMUP_ITERS,
            warmup_method=cfg.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
