# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.steering.methods.base import SteeringOperator
from vllm.steering.ops.triton_ops.steering_add_op import steering_add


class AdditiveSteeringOperator(SteeringOperator):
    """Additive activation steering.

    Applies x[i] += steering_vectors[indices[i]] * strengths[i]
    via the fused Triton kernel registered in Phase 1.
    """

    def apply(
        self,
        x: torch.Tensor,
        steering_vectors: torch.Tensor,
        indices: torch.Tensor,
        strengths: torch.Tensor,
    ) -> None:
        steering_add(x, steering_vectors, indices, strengths)
