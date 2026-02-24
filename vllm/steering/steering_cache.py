# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class SteeringCache:
    """Pre-allocated GPU cache for steering vectors.

    Manages a fixed-size tensor of shape [max_slots, hidden_dim] on the
    target device. Slots are allocated and released via a free-list,
    avoiding per-request memory allocation during inference.

    Design assumption: the cache is layer-agnostic. Every layer that
    applies steering indexes into the same [max_slots, hidden_dim]
    tensor. This matches the CAA (Contrastive Activation Addition)
    literature where the same vector is injected at every target
    layer. 
    
    TODO: If a future steering method requires per-layer vectors,
    the cache shape would need a layer dimension or separate caches
    per layer.
    """

    def __init__(
        self,
        max_slots: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self._max_slots = max_slots
        self._hidden_dim = hidden_dim
        self._cache = torch.zeros(
            max_slots, hidden_dim, dtype=dtype, device=device
        )
        self._free: set[int] = set(range(max_slots))

    def allocate(self) -> int:
        """Allocate a free slot and return its index.

        Raises:
            RuntimeError: If all slots are occupied.
        """
        if not self._free:
            raise RuntimeError(
                f"SteeringCache is full ({self._max_slots}/{self._max_slots} "
                f"slots occupied). Release a slot before allocating."
            )
        return self._free.pop()

    def release(self, slot: int) -> None:
        """Return a previously allocated slot to the free pool.

        Args:
            slot: Slot index to release.

        Raises:
            ValueError: If the slot index is out of range or not
                currently allocated.
        """
        if not (0 <= slot < self._max_slots):
            raise ValueError(
                f"Slot {slot} is out of range [0, {self._max_slots})"
            )
        if slot in self._free:
            raise ValueError(
                f"Slot {slot} is not currently allocated and cannot be "
                f"released."
            )
        self._free.add(slot)

    def store(self, slot: int, vector: torch.Tensor) -> None:
        """Copy a steering vector into the given cache slot.

        Args:
            slot: Target slot index (must be previously allocated).
            vector: Steering vector of shape [hidden_dim].

        Raises:
            ValueError: If the slot is not allocated or the vector
                shape does not match hidden_dim.
        """
        if slot in self._free or not (0 <= slot < self._max_slots):
            raise ValueError(
                f"Slot {slot} is not currently allocated"
            )
        if vector.shape != (self._hidden_dim,):
            raise ValueError(
                f"Expected vector of shape ({self._hidden_dim},), "
                f"got {tuple(vector.shape)}"
            )
        self._cache[slot].copy_(vector)

    @property
    def cache(self) -> torch.Tensor:
        """The underlying [max_slots, hidden_dim] tensor for kernel use."""
        return self._cache

    @property
    def num_free(self) -> int:
        """Number of available slots."""
        return len(self._free)

    @property
    def capacity(self) -> int:
        """Total number of slots."""
        return self._max_slots
