# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _steering_add_kernel(
    x_ptr,
    sv_ptr,
    indices_ptr,
    strengths_ptr,
    H,
    x_stride0,
    x_stride1,
    sv_stride0,
    sv_stride1,
    BLOCK_H: tl.constexpr,
):
    """Triton kernel for per-token additive activation steering.

    Performs x[i] += steering_vectors[indices[i]] * strengths[i].
    Tokens with index == -1 are skipped, ensuring zero overhead
    for unsteered requests and CUDA graph compatibility.

    Grid: (num_tokens, cdiv(H, BLOCK_H))
    """
    token_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)

    slot = tl.load(indices_ptr + token_id)

    if slot == -1:
        return

    strength = tl.load(strengths_ptr + token_id)

    col_offsets = block_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = col_offsets < H

    x_ptrs = x_ptr + token_id * x_stride0 + col_offsets * x_stride1
    sv_ptrs = sv_ptr + slot * sv_stride0 + col_offsets * sv_stride1

    x_vals = tl.load(x_ptrs, mask=mask)
    sv_vals = tl.load(sv_ptrs, mask=mask)

    x_vals = x_vals + sv_vals * strength

    tl.store(x_ptrs, x_vals, mask=mask)


@torch.inference_mode()
def _steering_add(
    x: torch.Tensor,
    steering_vectors: torch.Tensor,
    indices: torch.Tensor,
    strengths: torch.Tensor,
) -> None:
    """Applies additive steering in-place.

    For each token i, if indices[i] != -1:
        x[i] += steering_vectors[indices[i]] * strengths[i]

    Registered as torch.ops.vllm.steering_add so that
    torch.compile / CUDA graphs treat it as an opaque node.

    Args:
        x: Hidden states [B, H] (mutated in-place).
        steering_vectors: Steering vector cache [S, H].
        indices: Per-token slot index [B] (-1 indicates no steering).
        strengths: Per-token steering strength [B].
    """
    B, H = x.shape

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

    if B == 0:
        return

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
