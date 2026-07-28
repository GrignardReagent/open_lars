"""Functional core of the LARS optimizer.

The trust ratio ("adaptive learning rate") is just a handful of scalar
operations, so it is written as plain tensor ops -- nothing to compile, and
it runs on CPU or any CUDA/ROCm/MPS device torch supports, in any floating
dtype.
"""

import torch

__all__ = ["compute_adaptive_lr"]


def compute_adaptive_lr(param_norm, grad_norm, weight_decay, eps, trust_coef,
                        out=None):
    """Compute the LARS local ("adaptive") learning rate for one parameter.

    Implements the trust ratio of You, Gitman & Ginsburg (2017)::

        adaptive_lr = trust_coef * ||w|| / (||g|| + weight_decay * ||w|| + eps)

    with two guard cases:

    1. ``param_norm == 0`` (e.g. a parameter initialised to zero): the ratio
       would be zero and freeze the parameter forever, so fall back to 1.0
       (plain SGD-style update).
    2. ``grad_norm == 0`` (no useful gradient signal): the ratio is
       meaningless, so fall back to 1.0 and let the base update rule decide.

    Args:
        param_norm (Tensor): scalar tensor, ||w|| of the parameter.
        grad_norm (Tensor): scalar tensor, ||g|| of its (raw) gradient.
        weight_decay (float): weight-decay coefficient folded into the
            denominator (the paper's ``beta * ||w||`` term).
        eps (float): small constant for numerical stability.
        trust_coef (float): the paper's trust coefficient ``eta``.
        out (Tensor, optional): tensor to write the result into.

    Returns:
        Tensor: scalar tensor holding the adaptive lr (``out`` if given).
    """
    adaptive_lr = torch.where(
        (param_norm > 0) & (grad_norm > 0),
        trust_coef * param_norm / (grad_norm + weight_decay * param_norm + eps),
        torch.ones_like(param_norm),
    )
    if out is not None:
        out.copy_(adaptive_lr)
        return out
    return adaptive_lr
