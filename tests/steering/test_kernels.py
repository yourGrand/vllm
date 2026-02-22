# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.steering.ops.steering_op import _steering_add

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
HIDDEN_SIZES = [128, 1024, 4096]


def reference_steering_add(
    x: torch.Tensor,
    steering_vectors: torch.Tensor,
    indices: torch.Tensor,
    strengths: torch.Tensor,
) -> torch.Tensor:
    out = x.clone()
    mask = indices >= 0
    if mask.any():
        valid_indices = indices[mask]
        sv = steering_vectors[valid_indices]
        s = strengths[mask].unsqueeze(-1)
        out[mask] += sv * s
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
def test_correctness_all_steered(dtype, H):
    B, S = 32, 4
    torch.manual_seed(42)

    x = torch.randn(B, H, dtype=dtype, device="cuda")
    sv = torch.randn(S, H, dtype=dtype, device="cuda")
    indices = torch.randint(0, S, (B,), dtype=torch.int32, device="cuda")
    strengths = torch.ones(B, dtype=dtype, device="cuda")

    x_ref = reference_steering_add(x, sv, indices, strengths)

    _steering_add(x, sv, indices, strengths)

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    assert torch.allclose(x, x_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_masking_no_steering(dtype):
    B, H, S = 16, 512, 4
    torch.manual_seed(7)

    x = torch.randn(B, H, dtype=dtype, device="cuda")
    sv = torch.randn(S, H, dtype=dtype, device="cuda")
    indices = torch.full((B,), -1, dtype=torch.int32, device="cuda")
    strengths = torch.ones(B, dtype=dtype, device="cuda")

    x_orig = x.clone()

    _steering_add(x, sv, indices, strengths)

    assert torch.equal(x, x_orig)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_batch(dtype):
    B, H, S = 8, 256, 3
    torch.manual_seed(13)

    x = torch.randn(B, H, dtype=dtype, device="cuda")
    sv = torch.randn(S, H, dtype=dtype, device="cuda")

    indices = torch.tensor(
        [0, -1, 1, -1, 2, -1, 0, -1], dtype=torch.int32, device="cuda"
    )
    strengths = torch.ones(B, dtype=dtype, device="cuda")

    x_ref = reference_steering_add(x, sv, indices, strengths)

    _steering_add(x, sv, indices, strengths)

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    assert torch.allclose(x, x_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_per_token_strength(dtype):
    B, H, S = 16, 512, 2
    torch.manual_seed(99)

    x = torch.randn(B, H, dtype=dtype, device="cuda")
    sv = torch.randn(S, H, dtype=dtype, device="cuda")
    indices = torch.randint(0, S, (B,), dtype=torch.int32, device="cuda")
    strengths = torch.rand(B, dtype=dtype, device="cuda") * 2.0 - 1.0

    x_ref = reference_steering_add(x, sv, indices, strengths)

    _steering_add(x, sv, indices, strengths)

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    assert torch.allclose(x, x_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_multiple_vectors(dtype):
    B, H, S = 32, 1024, 8
    torch.manual_seed(2024)

    x = torch.randn(B, H, dtype=dtype, device="cuda")
    sv = torch.randn(S, H, dtype=dtype, device="cuda")
    indices = torch.randint(0, S, (B,), dtype=torch.int32, device="cuda")
    strengths = torch.full((B,), 0.5, dtype=dtype, device="cuda")

    x_ref = reference_steering_add(x, sv, indices, strengths)

    _steering_add(x, sv, indices, strengths)

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    assert torch.allclose(x, x_ref, atol=atol, rtol=rtol)


def test_zero_batch():
    H, S = 256, 4
    x = torch.randn(0, H, dtype=torch.float32, device="cuda")
    sv = torch.randn(S, H, dtype=torch.float32, device="cuda")
    indices = torch.zeros(0, dtype=torch.int32, device="cuda")
    strengths = torch.zeros(0, dtype=torch.float32, device="cuda")

    _steering_add(x, sv, indices, strengths)
    assert x.shape == (0, H)
