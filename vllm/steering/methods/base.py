# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch


class SteeringOperator(ABC):
    """Abstract base class for steering method implementations.

    Each steering method (additive, clamping, affine, etc.) subclasses
    this and implements :meth: apply, which mutates hidden states
    in-place using the shared compute primitives from vllm.steering.ops.
    """

    @abstractmethod
    def apply(
        self,
        x: torch.Tensor,
        steering_vectors: torch.Tensor,
        indices: torch.Tensor,
        strengths: torch.Tensor,
    ) -> None:
        """Apply the steering operation in-place on hidden states.

        Args:
            x: Hidden states [B, H], mutated in-place.
            steering_vectors: Cached steering vectors [S, H].
            indices: Per-token slot index [B].
                     A value of -1 means the token is not steered.
            strengths: Per-token steering strength [B].
        """
        ...
