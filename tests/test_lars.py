"""Behavioural tests for open_lars.LARS.

The centrepiece is the equivalence suite: open_lars.LARS must produce
*bit-identical* parameter trajectories to the reference implementation used
by IY036 (vendored verbatim in tests/reference_lars.py), on CPU and CUDA,
with and without weight decay / momentum, for both >1-D parameters (trust
ratio applied) and 1-D parameters (plain SGD+momentum fallback).
"""

import copy
import math

import pytest
import torch
from torch import nn

from open_lars import LARS
from tests.reference_lars import LARS as ReferenceLARS

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def make_model(device):
    # Mix of 2-D weights (trust ratio applied), biases and LayerNorm weights
    # (1-D: plain SGD+momentum), mirroring the transformer trained in IY036.
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.LayerNorm(16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).to(device)


def train_steps(model, optimizer, device, n_steps=10, scheduler=None, seed=0):
    torch.manual_seed(seed)
    for _ in range(n_steps):
        x = torch.randn(32, 8, device=device)
        y = torch.randn(32, 4, device=device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return loss


# ── Equivalence with the IY036 reference implementation ──────────────────────

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('momentum,weight_decay', [
    (0.9, 1e-4),   # IY036's configuration
    (0.9, 0.0),
    (0.0, 1e-4),
    (0.0, 0.0),
])
def test_matches_reference_implementation(device, momentum, weight_decay):
    torch.manual_seed(42)
    model_a = make_model(device)
    model_b = copy.deepcopy(model_a)

    opt_a = LARS(model_a.parameters(), lr=0.5, momentum=momentum,
                 weight_decay=weight_decay, trust_coefficient=0.001)
    opt_b = ReferenceLARS(model_b.parameters(), lr=0.5, momentum=momentum,
                          weight_decay=weight_decay, trust_coefficient=0.001)

    train_steps(model_a, opt_a, device, n_steps=10)
    train_steps(model_b, opt_b, device, n_steps=10)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


# ── Update-rule unit checks ──────────────────────────────────────────────────

def test_single_step_matches_formula_2d():
    torch.manual_seed(0)
    p = torch.randn(4, 4, requires_grad=True)
    p_init = p.detach().clone()
    lr, wd, tc, eps = 0.1, 1e-4, 0.001, 1e-8

    loss = (p ** 2).sum()
    loss.backward()
    grad = p.grad.detach().clone()

    LARS([p], lr=lr, momentum=0.9, weight_decay=wd, trust_coefficient=tc).step()

    w_norm = p_init.norm()
    g_norm = grad.norm()
    local_lr = (tc * w_norm / (g_norm + wd * w_norm + eps)).item()
    expected = p_init - lr * local_lr * (grad + wd * p_init)
    assert torch.allclose(p.detach(), expected, atol=1e-7)


def test_1d_params_use_plain_sgd_momentum():
    # A bias-like 1-D parameter must be updated with local_lr == 1.0.
    torch.manual_seed(0)
    p = torch.randn(6, requires_grad=True)
    p_init = p.detach().clone()
    lr = 0.01

    (p ** 2).sum().backward()
    grad = p.grad.detach().clone()

    LARS([p], lr=lr, momentum=0.9, weight_decay=0.0).step()

    assert torch.allclose(p.detach(), p_init - lr * grad, atol=1e-7)


def test_momentum_accumulates_velocity():
    # Two steps with a constant gradient: v2 = m*v1 + lr*g, w2 = w0 - v1 - v2.
    p = torch.ones(3, 3, requires_grad=True)
    lr, m, tc, eps = 0.1, 0.9, 0.001, 1e-8
    grad = torch.full((3, 3), 0.5)

    opt = LARS([p], lr=lr, momentum=m, weight_decay=0.0, trust_coefficient=tc)

    w = p.detach().clone()
    v = torch.zeros_like(w)
    for _ in range(2):
        local_lr = (tc * w.norm() / (grad.norm() + eps)).item()
        v = m * v + lr * local_lr * grad
        w = w - v
        p.grad = grad.clone()
        opt.step()

    assert torch.allclose(p.detach(), w, atol=1e-7)


def test_zero_weight_param_still_updates():
    # ||w|| == 0 would give trust ratio 0 and freeze the parameter forever;
    # the guard must fall back to local_lr = 1.0 instead.
    p = torch.zeros(4, 4, requires_grad=True)
    p.grad = torch.ones(4, 4)

    LARS([p], lr=0.1, momentum=0.0).step()

    assert torch.allclose(p.detach(), torch.full((4, 4), -0.1))


def test_zero_grad_is_a_noop():
    p = torch.randn(4, 4, requires_grad=True)
    p_init = p.detach().clone()
    p.grad = torch.zeros(4, 4)

    LARS([p], lr=0.1, momentum=0.9, weight_decay=0.0).step()

    assert torch.equal(p.detach(), p_init)


# ── torch.optim.Optimizer contract ───────────────────────────────────────────

def test_closure_returns_loss():
    p = torch.randn(3, 3, requires_grad=True)
    opt = LARS([p], lr=0.1)

    def closure():
        opt.zero_grad()
        loss = (p ** 2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None and loss.item() >= 0.0


@pytest.mark.parametrize('device', DEVICES)
def test_state_dict_roundtrip(device):
    torch.manual_seed(1)
    model_a = make_model(device)
    opt_a = LARS(model_a.parameters(), lr=0.5, weight_decay=1e-4)
    train_steps(model_a, opt_a, device, n_steps=5)

    # Resume from a checkpoint into fresh objects. The deepcopy stands in for
    # a torch.save/torch.load round trip -- loading a live state_dict directly
    # would alias opt_a's momentum buffers.
    model_b = copy.deepcopy(model_a)
    opt_b = LARS(model_b.parameters(), lr=0.5, weight_decay=1e-4)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))

    # ...and both must continue on the exact same trajectory.
    train_steps(model_a, opt_a, device, n_steps=5, seed=2)
    train_steps(model_b, opt_b, device, n_steps=5, seed=2)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_lr_scheduler_compatibility():
    # IY036 pairs LARS with linear-warmup + cosine decay
    # (transformers.get_cosine_schedule_with_warmup); reproduce that schedule
    # with a plain LambdaLR to keep this package dependency-free.
    torch.manual_seed(3)
    model = make_model('cpu')
    opt = LARS(model.parameters(), lr=1.0, weight_decay=1e-4)

    total, warmup = 20, 5

    def warmup_cosine(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine)

    train_steps(model, opt, 'cpu', n_steps=total, scheduler=scheduler)
    assert opt.param_groups[0]['lr'] == pytest.approx(0.0, abs=1e-6)


def test_param_groups_with_distinct_hyperparams():
    w = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    opt = LARS([
        {'params': [w], 'weight_decay': 1e-4},
        {'params': [b], 'weight_decay': 0.0, 'lr': 0.01},
    ], lr=0.5)

    assert opt.param_groups[0]['weight_decay'] == 1e-4
    assert opt.param_groups[1]['lr'] == 0.01

    w.grad = torch.ones_like(w)
    b.grad = torch.ones_like(b)
    opt.step()  # must not raise


def test_invalid_hyperparams_raise():
    p = [torch.zeros(2, 2)]
    with pytest.raises(ValueError):
        LARS(p, lr=-0.1)
    with pytest.raises(ValueError):
        LARS(p, lr=0.1, momentum=-0.5)
    with pytest.raises(ValueError):
        LARS(p, lr=0.1, weight_decay=-1e-4)
    with pytest.raises(ValueError):
        LARS(p, lr=0.1, trust_coefficient=-0.001)
    with pytest.raises(ValueError):
        LARS(p, lr=0.1, eps=-1e-8)


# ── End-to-end sanity ────────────────────────────────────────────────────────

@pytest.mark.parametrize('device', DEVICES)
def test_converges_on_toy_regression(device):
    torch.manual_seed(7)
    true_w = torch.randn(8, 1, device=device)
    x = torch.randn(256, 8, device=device)
    y = x @ true_w

    model = nn.Linear(8, 1, bias=False).to(device)
    opt = LARS(model.parameters(), lr=5.0, momentum=0.9,
               trust_coefficient=0.01)

    initial_loss = nn.functional.mse_loss(model(x), y).item()
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.05 * initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_cuda_matches_cpu_trajectory():
    torch.manual_seed(11)
    model_cpu = make_model('cpu')
    model_gpu = copy.deepcopy(model_cpu).cuda()

    opt_cpu = LARS(model_cpu.parameters(), lr=0.5, weight_decay=1e-4)
    opt_gpu = LARS(model_gpu.parameters(), lr=0.5, weight_decay=1e-4)

    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    for _ in range(5):
        for model, opt, xd, yd in [(model_cpu, opt_cpu, x, y),
                                   (model_gpu, opt_gpu, x.cuda(), y.cuda())]:
            opt.zero_grad()
            nn.functional.mse_loss(model(xd), yd).backward()
            opt.step()

    for p_c, p_g in zip(model_cpu.parameters(), model_gpu.parameters()):
        assert torch.allclose(p_c, p_g.cpu(), atol=1e-5)
