# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.steering.ops.triton_ops.kernels import _steering_add_kernel
from vllm.triton_utils import triton
from vllm.utils.torch_utils import direct_register_custom_op


@torch.inference_mode()
def _steering_add(
    x: torch.Tensor,
    steering_vectors: torch.Tensor,
    indices: torch.Tensor,
    strengths: torch.Tensor,
) -> None:
    """
    Applies additive steering in-place:
        for each token i:
            if indices[i] != -1:
                x[i] += steering_vectors[indices[i]] * strengths[i]

    This function is registered as torch.ops.vllm.steering_add so that
    torch.compile / CUDA graphs treat it as an opaque node whose topology
    never changes.

    Args:
        x: Hidden states [B, H] (mutated in-place)
        steering_vectors: Steering vector cache [S, H]
        indices: Per-token slot index [B] (-1 indicates no steering)
        strengths: Per-token steering strength [B]
    """
    B, H = x.shape

    # Validation
    assert x.ndim == 2, f"x must be 2D [B, H], got {x.shape}"
    assert steering_vectors.ndim == 2, (
        f"steering_vectors must be 2D [S, H], got {steering_vectors.shape}"
    )
    assert steering_vectors.shape[1] == H, (
        f"Hidden dim mismatch: x has H={H}, "
        f"steering_vectors has H={steering_vectors.shape[1]}"
    )
    assert indices.shape == (B,), (
        f"indices must be [B={B}], got {indices.shape}"
    )
    assert strengths.shape == (B,), (
        f"strengths must be [B={B}], got {strengths.shape}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    assert steering_vectors.is_contiguous(), (
        "steering_vectors must be contiguous"
    )

    # Early exit: no tokens need steering
    if B == 0:
        return

    # Kernel launch
    BLOCK_H = triton.next_power_of_2(min(H, 1024))

    grid = (B, triton.cdiv(H, BLOCK_H))

    _steering_add_kernel[grid](
        x,
        steering_vectors,
        indices,
        strengths,
        H,
        x.stride(0),
        x.stride(1),
        steering_vectors.stride(0),
        steering_vectors.stride(1),
        BLOCK_H=BLOCK_H,
    )


def _steering_add_fake(
    x: torch.Tensor,
    steering_vectors: torch.Tensor,
    indices: torch.Tensor,
    strengths: torch.Tensor,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="steering_add",
        op_func=_steering_add,
        mutates_args=["x"],
        fake_impl=_steering_add_fake,
    )
    steering_add = torch.ops.vllm.steering_add

except AttributeError:
    steering_add = _steering_add