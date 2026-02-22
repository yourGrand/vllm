# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Type

from vllm.steering.methods.base import SteeringOperator

_REGISTRY: dict[str, Type[SteeringOperator]] = {}


def register_operator(name: str, cls: Type[SteeringOperator]) -> None:
    """Register a steering operator class under the given name.

    Args:
        name: Lookup key (e.g. additive).
        cls: A concrete :class: SteeringOperator subclass.
    """
    _REGISTRY[name] = cls


def get_operator(name: str) -> SteeringOperator:
    """Instantiate and return the steering operator registered under *name*.

    Args:
        name: Method name previously passed to :func: register_operator.

    Returns:
        A new instance of the corresponding :class: SteeringOperator.

    Raises:
        ValueError: If name has not been registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown steering method {name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()
