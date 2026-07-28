"""Tests for open_lars.compute_adaptive_lr.

The function is pure tensor ops, so every case is checked across floating
dtypes and (when available) across devices, including half precision.
"""

import pytest
import torch

from open_lars import compute_adaptive_lr

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
DTYPES = [torch.float, torch.double]


def expected_ratio(param_norm, grad_norm, weight_decay, eps, trust_coef):
    return trust_coef * param_norm / (grad_norm + weight_decay * param_norm + eps)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('param_norm,grad_norm,weight_decay,eps,trust_coef', [
    (2.5, 0.5, 1e-4, 1e-8, 0.001),
    (0.037, 12.0, 1e-4, 1e-8, 0.001),
    (1.0, 1.0, 1.0, 2.0, 1.0),
    (3.0, 0.25, 0.0, 0.0, 0.02),
])
def test_matches_formula(device, dtype, param_norm, grad_norm, weight_decay,
                         eps, trust_coef):
    result = compute_adaptive_lr(
        torch.tensor(param_norm, dtype=dtype, device=device),
        torch.tensor(grad_norm, dtype=dtype, device=device),
        weight_decay, eps, trust_coef)

    expected = expected_ratio(param_norm, grad_norm, weight_decay, eps, trust_coef)
    assert torch.allclose(result, torch.tensor(expected, dtype=dtype, device=device))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_zero_param_norm_falls_back_to_one(device, dtype):
    # ||w|| == 0 would give a zero ratio and freeze the parameter forever.
    result = compute_adaptive_lr(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(1., dtype=dtype, device=device),
        1e-4, 1e-8, 0.001)

    assert result == torch.tensor(1., dtype=dtype, device=device)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_zero_grad_norm_falls_back_to_one(device, dtype):
    # ||g|| == 0 carries no signal; defer to the base update rule.
    result = compute_adaptive_lr(
        torch.tensor(1., dtype=dtype, device=device),
        torch.tensor(0., dtype=dtype, device=device),
        1e-4, 1e-8, 0.001)

    assert result == torch.tensor(1., dtype=dtype, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
@pytest.mark.parametrize('dtype', DTYPES)
def test_cpu_and_gpu_agree(dtype):
    args = (0.8, 3.2, 1e-4, 1e-8, 0.001)

    cpu = compute_adaptive_lr(
        torch.tensor(args[0], dtype=dtype),
        torch.tensor(args[1], dtype=dtype), *args[2:])
    gpu = compute_adaptive_lr(
        torch.tensor(args[0], dtype=dtype, device='cuda'),
        torch.tensor(args[1], dtype=dtype, device='cuda'), *args[2:])

    assert torch.allclose(cpu, gpu.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
@pytest.mark.parametrize('param_norm,grad_norm,expected', [
    (2.5, 0.5, 0.001 * 2.5 / (0.5 + 1e-4 * 2.5 + 1e-8)),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
])
def test_half_precision_on_gpu(param_norm, grad_norm, expected):
    result = compute_adaptive_lr(
        torch.tensor(param_norm, dtype=torch.half, device='cuda'),
        torch.tensor(grad_norm, dtype=torch.half, device='cuda'),
        1e-4, 1e-8, 0.001)

    assert torch.allclose(result, torch.tensor(expected, dtype=torch.half, device='cuda'))


@pytest.mark.parametrize('dtype', DTYPES)
def test_out_argument_is_written_in_place(dtype):
    out = torch.tensor(0., dtype=dtype)

    returned = compute_adaptive_lr(
        torch.tensor(2.5, dtype=dtype),
        torch.tensor(0.5, dtype=dtype),
        1e-4, 1e-8, 0.001, out)

    assert returned is out
    expected = expected_ratio(2.5, 0.5, 1e-4, 1e-8, 0.001)
    assert torch.allclose(out, torch.tensor(expected, dtype=dtype))
