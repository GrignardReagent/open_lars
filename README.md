# open_lars

A **pure-PyTorch implementation of the LARS optimizer** (Layer-wise Adaptive
Rate Scaling; [You, Gitman & Ginsburg 2017](https://arxiv.org/abs/1708.03888)).

LARS is the optimizer used by the SimCLR / SupCon lineage (Chen et al. 2020;
Khosla et al. 2020) for large-batch contrastive pretraining: at very large
batch sizes a single global learning rate is simultaneously too large for some
layers and too small for others, so LARS rescales each layer's effective LR by
a "trust ratio" - the ratio of that layer's weight norm to its gradient norm.

## Why pure PyTorch?

Older LARS packages computed the trust ratio in custom C++/CUDA kernels
compiled at install time. That approach ages badly: it needs an `nvcc`
matching the exact CUDA version your PyTorch wheel was built against (fragile
on shared or HPC machines, impossible where no CUDA toolkit is installed), it
pins you to one torch ABI so every torch upgrade means a rebuild, and once the
underlying C++/ATen APIs churn, the extension stops building altogether.

The trust ratio is a handful of scalar operations, so `open_lars` implements
it as plain tensor ops instead. There is nothing to compile: it runs on CPU
and on **any CUDA version supported by your PyTorch build** (also ROCm/MPS),
with any `torch >= 1.8`.

## Install

```bash
pip install open_lars
```

Or from source:

```bash
pip install .            # from a checkout
pip install -e .         # editable, for development
```

The only dependency is `torch`. Once installed in an environment, it is
importable from any directory you work in.

## Usage

`open_lars.LARS` is a standalone optimizer (not a wrapper around another
optimizer), with SGD-with-momentum as the inner update rule, exactly as in
the paper's Algorithm 1:

```python
from open_lars import LARS

optimizer = LARS(model.parameters(), lr=1.6, momentum=0.9,
                 weight_decay=1e-4, trust_coefficient=0.001)

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

It is a standard `torch.optim.Optimizer`: parameter groups, `state_dict()` /
`load_state_dict()` checkpointing, closures, and LR schedulers (e.g.
linear-warmup + cosine decay, the schedule conventionally paired with LARS)
all work as usual.

### Update rule

```text
local_lr = trust_coefficient * ||w|| / (||∇L(w)|| + weight_decay * ||w|| + eps)
v_t      = momentum * v_{t-1} + lr * local_lr * (∇L(w) + weight_decay * w)
w_{t+1}  = w_t - v_t
```

Following standard practice (the paper; SimCLR/BYOL reference code), the trust
ratio is **skipped for 1-D parameters** (biases, LayerNorm/BatchNorm weights),
which fall back to plain SGD+momentum — applying per-layer adaptation to those
is known to hurt convergence. If `||w||` or `||∇L(w)||` is zero, `local_lr`
falls back to `1.0` so the parameter still receives a plain update.

The trust-ratio computation is also exposed directly as a pure-torch function:

```python
from open_lars import compute_adaptive_lr
```

## Tests

```bash
pip install -e '.[test]'
pytest
```

The suite covers the trust-ratio formula and its zero-norm guards across
dtypes and devices (including half precision), the optimizer contract
(pickling, checkpointing, schedulers, parameter groups, closures), and an
equivalence suite verifying that `open_lars.LARS` reproduces the reference
implementation in `tests/reference_lars.py` **bit-for-bit** over multi-step
training trajectories, on CPU and CUDA, across momentum/weight-decay
configurations. CUDA-specific tests skip automatically on CPU-only machines.

## References

- You, Gitman & Ginsburg, *Large Batch Training of Convolutional Networks*,
  2017. <https://arxiv.org/abs/1708.03888>
- Chen et al., *A Simple Framework for Contrastive Learning of Visual
  Representations* (SimCLR), 2020. <https://arxiv.org/abs/2002.05709>
- Khosla et al., *Supervised Contrastive Learning*, 2020.
  <https://arxiv.org/abs/2004.11362>

## License

Apache License 2.0 — see `LICENSE`.
