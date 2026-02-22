# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.steering.methods.additive.operator import AdditiveSteeringOperator
from vllm.steering.methods.registry import register_operator

register_operator("additive", AdditiveSteeringOperator)

__all__ = ["AdditiveSteeringOperator"]
