"""
LARS (Layer-wise Adaptive Rate Scaling) optimizer, in pure PyTorch.

Reference
---------
You, Gitman & Ginsburg, "Large Batch Training of Convolutional Networks", 2017.
https://arxiv.org/abs/1708.03888

Why this exists: at very large batch sizes, a single global learning rate is
simultaneously too large for some layers (divergence) and too small for
others (slow convergence), because layers differ widely in parameter/gradient
scale. LARS rescales each layer's *effective* LR by a "trust ratio" -- the
ratio of that layer's weight norm to its gradient norm -- so every layer
takes a step proportional to its own parameter scale rather than one shared
LR. This is the optimizer Khosla et al. (SupCon) and the SimCLR lineage use
for large-batch contrastive pretraining.

Implements the paper's Algorithm 1 directly:
    v_t   = momentum * v_{t-1}  +  lr * local_lr * (grad + weight_decay * w)
    w_t+1 = w_t - v_t
    local_lr = trust_coefficient * ||w|| / (||grad|| + weight_decay * ||w|| + eps)

There is no custom CUDA kernel to compile at install time: the trust ratio is
plain tensor ops (see ``open_lars.functional.compute_adaptive_lr``), so this
runs against any current CUDA build of PyTorch -- or CPU/ROCm/MPS -- with no
nvcc and no ABI pinning.

Following standard practice (the paper; also SimCLR/BYOL reference code),
LARS scaling is skipped for 1-D parameters (biases, LayerNorm/BatchNorm
weights) -- applying a per-layer trust ratio to those is known to hurt
convergence -- and falls back to plain SGD+momentum for them instead.
"""

import torch
from torch.optim import Optimizer

from open_lars.functional import compute_adaptive_lr

__all__ = ["LARS"]


class LARS(Optimizer):
    """LARS as a self-contained :class:`torch.optim.Optimizer`.

    LARS *is* the optimizer here (rather than a wrapper around another one),
    with SGD-with-momentum as the inner update rule, exactly as in the
    paper's Algorithm 1.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        lr (float): global learning rate.
        momentum (float, optional): SGD-style momentum used inside LARS.
        weight_decay (float, optional): L2 penalty, folded into both the
            trust-ratio denominator and the update direction.
        trust_coefficient (float, optional): the paper's ``eta``.
        eps (float, optional): numerical-stability constant in the
            trust-ratio denominator.

    Example::

        optimizer = LARS(model.parameters(), lr=1.6, momentum=0.9,
                         weight_decay=1e-4, trust_coefficient=0.001)

        loss = loss_fn(model(input), target)
        loss.backward()
        optimizer.step()
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0,
                 trust_coefficient=0.001, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust_coefficient value: {trust_coefficient}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            trust_coefficient = group["trust_coefficient"]
            eps = group["eps"]
            global_lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                # Trust ratio uses the RAW gradient norm in its denominator
                # (matches the paper's ||∇L(w)|| term) -- weight decay is
                # folded in separately, only for the velocity update itself.
                if p.ndim > 1:
                    local_lr = compute_adaptive_lr(
                        torch.norm(p), torch.norm(grad),
                        weight_decay, eps, trust_coefficient).item()
                else:
                    local_lr = 1.0  # biases / norm params: plain SGD+momentum, no trust ratio

                d_p = grad if weight_decay == 0 else grad.add(p, alpha=weight_decay)
                actual_lr = global_lr * local_lr

                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = d_p.mul(actual_lr)
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=actual_lr)

                p.add_(buf, alpha=-1.0)

        return loss
