# SPDX-License-Identifier: Apache-2.0
"""Standalone Kunlun-aware ``bind_kv_cache`` replacement.

Isolated from ``vllm_kunlun.v1.worker.utils`` so that patching does not drag
in heavy imports (e.g. ``vllm.attention.backends.abstract``) whose layout
differs between vLLM versions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.attention.layer import Attention


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: Optional[int] = 1,
) -> None:
    """Kunlun-aware replacement for ``vllm.v1.worker.utils.bind_kv_cache``.

    Identical to upstream except that ``is_kunlun()`` is accepted alongside
    CUDA/XPU when multiple attention layers share a single ``layer_index``
    (as happens for encoder-decoder or MTP drafter stacks).
    """
    assert len(runner_kv_caches) == 0

    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            if (
                current_platform.is_kunlun()
                or current_platform.is_cuda()
                or current_platform.is_xpu()
            ):
                pass
            else:
                raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = [kv_cache]
