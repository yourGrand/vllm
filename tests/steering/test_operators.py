# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.steering.methods.additive  # noqa: F401
from vllm.steering.methods.additive import AdditiveSteeringOperator
from vllm.steering.methods.base import SteeringOperator
from vllm.steering.methods.registry import get_operator


def test_operator_abc_contract():
    """SteeringOperator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SteeringOperator()


def test_additive_operator_wiring():
    """AdditiveSteeringOperator.apply() produces correct output end-to-end."""
    B, H, S = 8, 256, 2
    torch.manual_seed(42)

    op = AdditiveSteeringOperator()

    x = torch.randn(B, H, dtype=torch.float32, device="cuda")
    sv = torch.randn(S, H, dtype=torch.float32, device="cuda")
    indices = torch.tensor(
        [0, -1, 1, -1, 0, -1, 1, -1], dtype=torch.int32, device="cuda"
    )
    strengths = torch.ones(B, dtype=torch.float32, device="cuda")

    x_orig = x.clone()

    op.apply(x, sv, indices, strengths)

    steered = indices >= 0
    assert not torch.equal(x[steered], x_orig[steered])
    assert torch.equal(x[~steered], x_orig[~steered])


def test_registry_returns_correct_operator():
    """get_operator('additive') returns an AdditiveSteeringOperator."""
    op = get_operator("additive")
    assert isinstance(op, AdditiveSteeringOperator)


def test_registry_unknown_method():
    """get_operator() raises ValueError for unregistered names."""
    with pytest.raises(ValueError, match="Unknown steering method"):
        get_operator("nonexistent")
