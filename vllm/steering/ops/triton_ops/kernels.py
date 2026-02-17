# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


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
    """
    Triton kernel for activation steering.

    Performs per-token additive steering:
        x[i] += steering_vectors[indices[i]] * strengths[i]

    Tokens with index == -1 are skipped (no-op), ensuring zero overhead
    for unsteered requests and CUDA graph compatibility.

    Args:
        x_ptr: Pointer to hidden states [B, H]
        sv_ptr: Pointer to steering vectors [S, H]
        indices_ptr: Pointer to slot indices [B]
        strengths_ptr: Pointer to strengths [B]
        H: Hidden size
        x_stride0, x_stride1: Strides for x
        sv_stride0, sv_stride1: Strides for steering vectors
        BLOCK_H: Block size for hidden dimension

    Grid: (num_tokens, cdiv(H, BLOCK_H))
    """
    token_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)

    slot = tl.load(indices_ptr + token_id)

    # Early exit: slot == -1 means this token is not steered.
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