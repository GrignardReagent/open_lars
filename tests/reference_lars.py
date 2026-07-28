"""Copy of the reference LARS implementation used by IY036
(stochastic_simulations/src/training/lars.py), vendored here as ground
truth for the equivalence tests in test_lars.py. Code is verbatim; only
the docstring differs. Do not edit the code."""

"""
LARS (Layer-wise Adaptive Rate Scaling) optimizer.

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

No custom CUDA kernel to compile at install time -- fragile on a shared or
HPC environment. This is plain tensor ops, so it runs anywhere torch does.

Following standard practice (the paper; also SimCLR/BYOL reference code),
LARS scaling is skipped for 1-D parameters (biases, LayerNorm/BatchNorm
weights) -- applying a per-layer trust ratio to those is known to hurt
convergence -- and falls back to plain SGD+momentum for them instead.
"""

import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0,
                 trust_coefficient=0.001, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
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

                # Trust ratio uses the RAW gradient norm in its denominator
                # (matches the paper's ||∇L(w)|| term) -- weight decay is
                # folded in separately, only for the velocity update itself.
                if p.ndim > 1:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(grad)
                    trust_ratio = torch.where(
                        (w_norm > 0) & (g_norm > 0),
                        trust_coefficient * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm),
                    )
                    local_lr = trust_ratio.item()
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
