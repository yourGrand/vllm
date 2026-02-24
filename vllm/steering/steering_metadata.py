# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class SteeringMetadata:
    """Pre-allocated metadata manager for per-token steering dispatch.

    Owns the indices and strengths tensors that the steering kernel
    consumes. Both tensors are allocated once at maximum batch size and
    updated in-place each step via update(), avoiding per-step
    allocations for CUDA graph compatibility.

    Architecturally mirrors PunicaWrapperGPU in the LoRA subsystem, but
    only requires two metadata tensors instead of four because steering
    operates on layer outputs (vector addition) rather than weights
    (matrix multiplication).

    Indices use int32 dtype throughout the steering subsystem.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        device: torch.device | str,
    ) -> None:
        self._max_num_batched_tokens = max_num_batched_tokens
        self._device = device
        self._num_tokens = 0

        self._indices = torch.full(
            (max_num_batched_tokens,),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        self._strengths = torch.zeros(
            max_num_batched_tokens,
            dtype=torch.float32,
            device=device,
        )

    # TODO(Phase 5): Replace dict args with a SteeringMapping dataclass
    # (analogous to LoRAMapping) for type-safe scheduler integration.
    # TODO(Phase 5): Batch writes via index_put_ or build CPU tensors
    # and copy_ to avoid per-element CUDA operations at scale.
    def update(
        self,
        token_to_slot: dict[int, int],
        token_to_strength: dict[int, float],
        num_tokens: int,
    ) -> None:
        """Update per-token steering metadata for the current batch.

        Resets the active region to defaults (index -1, strength 0.0),
        then writes the provided mappings. Tokens not present in the
        mappings remain unsteered.

        Args:
            token_to_slot: Maps token position to cache slot index.
            token_to_strength: Maps token position to steering strength.
            num_tokens: Number of active tokens in the current batch.

        Raises:
            ValueError: If num_tokens exceeds the pre-allocated capacity.
        """
        if num_tokens > self._max_num_batched_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds pre-allocated capacity "
                f"({self._max_num_batched_tokens})"
            )
        self._num_tokens = num_tokens

        self._indices[:num_tokens].fill_(-1)
        self._strengths[:num_tokens].fill_(0.0)

        for pos, slot in token_to_slot.items():
            self._indices[pos] = slot
        for pos, strength in token_to_strength.items():
            self._strengths[pos] = strength

    @property
    def indices(self) -> torch.Tensor:
        """Per-token slot indices for the active batch.

        Shape [num_tokens]. A value of -1 means the token is not steered.
        """
        return self._indices[: self._num_tokens]

    @property
    def strengths(self) -> torch.Tensor:
        """Per-token steering strengths for the active batch.

        Shape [num_tokens].
        """
        return self._strengths[: self._num_tokens]

    def reset(self) -> None:
        """Reset all metadata to default values."""
        self._indices.fill_(-1)
        self._strengths.fill_(0.0)
        self._num_tokens = 0
