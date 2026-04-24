# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kunlun XPU adaptation for vllm.v1.spec_decode.eagle.EagleProposer.

Design principle:
    Patch ONLY the parts that upstream implements with ops XPU cannot run
    (Triton kernels, index_fill_ on empty indices, etc.).
    DO NOT copy the entire `propose()` body - it pulls in too many upstream
    internals and breaks every time upstream refactors them.

Current patched members:
    - prepare_next_token_ids_padded:
        Upstream uses a Triton kernel; XPU has no Triton. We re-implement
        it with pure PyTorch ops and also avoid `index_fill_` on empty
        indices (XPU/XMLIR limitation) by using `torch.where` with a mask.
"""
from typing import Optional  # noqa: F401

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


def prepare_next_token_ids_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kunlun XPU replacement for EagleProposer.prepare_next_token_ids_padded.

    Upstream implements this with a Triton kernel; XPU has no Triton, so we
    re-implement with pure PyTorch ops. We also avoid `index_fill_` on empty
    index tensors (not supported on XPU/XMLIR) by using a boolean mask with
    `torch.where`.

    Semantics follow upstream:
      - For each request, pick the right-most "valid" sampled token.
      - A token is valid iff its row is not marked in `discard_request_mask`
        AND the token id is in [0, vocab_size).
      - If no valid token is found, fall back to the pre-computed "backup"
        next-token (obtained via `CachedRequestState.get_token_id`).
    """
    # Precompute get_token_id for when there is no valid next token.
    # NOTE: explicit dtype=np.int32 to match upstream; otherwise Python ints
    # default to int64 on 64-bit hosts, causing the final next_token_ids
    # tensor to inherit int64 instead of the int32 that self.input_ids
    # (and downstream draft model) expect.
    num_reqs = gpu_input_batch.num_reqs
    self.backup_next_token_ids.np[:num_reqs] = np.array(
        [
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                common_attn_metadata.seq_lens_cpu[i].item()
            )
            for i in range(num_reqs)
        ],
        dtype=np.int32,
    )
    self.backup_next_token_ids.copy_to_gpu(num_reqs)

    batch_size = sampled_token_ids.shape[0]

    # Per-row discard mask -> replace those rows' tokens with -1 so the
    # vocab-range check below fails and valid_count becomes 0, exactly like
    # upstream's Triton kernel which early-exits discarded rows to
    # valid_count=0 / next_token=backup.
    discard_mask = discard_request_mask[:batch_size].to(
        device=sampled_token_ids.device, dtype=torch.bool
    )
    valid_sampled_token_ids_gpu = torch.where(
        discard_mask.unsqueeze(1),
        torch.full_like(sampled_token_ids, -1),
        sampled_token_ids,
    )

    # Valid token mask: token id in [0, vocab_size). Unconditionally
    # apply the bounds check (upstream kernel does the same regardless
    # of max_gen_len); a discarded row is already filled with -1 and
    # will count 0 valid tokens.
    valid_mask = (valid_sampled_token_ids_gpu != -1) & (
        valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
    )

    valid_sampled_tokens_count = valid_mask.sum(dim=1)

    # Right-most valid index per row.
    last_valid_indices = valid_sampled_tokens_count - 1
    last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

    selected_tokens = torch.gather(
        valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
    ).squeeze(1)

    # Use last valid token, else pre-computed backup. Force int32 to
    # match upstream `torch.empty(..., dtype=torch.int32)` contract.
    backup_gpu = self.backup_next_token_ids.gpu[:batch_size].to(torch.int32)
    next_token_ids = torch.where(
        last_valid_indices != -1,
        selected_tokens.to(torch.int32),
        backup_gpu,
    )

    return next_token_ids, valid_sampled_tokens_count


# NOTE: We intentionally do NOT patch EagleProposer.propose. Upstream's
# implementation is the source of truth; kunlun only needs to override ops
# at lower levels (e.g. kunlun_attn backend, Triton-backed helpers).
EagleProposer.prepare_next_token_ids_padded = prepare_next_token_ids_padded


def prepare_inputs_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    spec_decode_metadata,  # SpecDecodeMetadata
    valid_sampled_tokens_count: torch.Tensor,
):
    """
    Kunlun XPU replacement for EagleProposer.prepare_inputs_padded.

    Upstream fuses per-request index computation into a Triton kernel
    (`eagle_prepare_inputs_padded_kernel`), which requires libcuda on XPU
    and therefore fails at driver init. We re-implement the kernel body
    with pure PyTorch ops (all vectorized over requests).
    """
    num_reqs = common_attn_metadata.num_reqs
    device = valid_sampled_tokens_count.device

    # cu_num_draft_tokens is an INCLUSIVE cumulative sum (len == num_reqs).
    cu_num_draft = spec_decode_metadata.cu_num_draft_tokens
    # num_draft_tokens[i] = cu[i] - cu[i-1], with prev = 0 for i == 0.
    num_draft = cu_num_draft.clone()
    if num_reqs > 1:
        num_draft[1:] = cu_num_draft[1:] - cu_num_draft[:-1]

    valid_count = valid_sampled_tokens_count.to(num_draft.dtype)
    num_rejected = num_draft + 1 - valid_count
    num_rejected = torch.where(
        num_draft > 0, num_rejected, torch.zeros_like(num_rejected)
    )

    # q_last_tok_idx[i] = query_start_loc[i + 1] - 1
    q_last_tok_idx = common_attn_metadata.query_start_loc[1:] - 1

    token_indices_to_sample = (q_last_tok_idx - num_rejected).to(torch.int32)
    num_rejected_tokens_gpu = num_rejected.to(torch.int32)

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    total_num_tokens = query_start_loc_cpu[-1].item()

    spec_common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=common_attn_metadata.query_start_loc,
        seq_lens=common_attn_metadata.seq_lens,
        query_start_loc_cpu=query_start_loc_cpu,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
        num_reqs=common_attn_metadata.num_reqs,
        num_actual_tokens=total_num_tokens,
        max_query_len=new_query_len_per_req.max().item(),
        max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
        causal=True,
        dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
    )

    return (
        spec_common_attn_metadata,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
    )


EagleProposer.prepare_inputs_padded = prepare_inputs_padded


# ---------------------------------------------------------------------------
# Workaround for an upstream vLLM bug in EagleProposer.propose (as of the
# version pinned by this repo):
#
#   if self.uses_mrope:
#       positions = self.positions[:, last_token_indices]   # <- wrong attr
#   else:
#       positions = self.positions[last_token_indices]
#
# When `uses_mrope` is True, `__init__` only creates `self.mrope_positions`
# and never creates `self.positions`, so the above line raises
# `AttributeError: 'EagleProposer' object has no attribute 'positions'`.
#
# This happens for any M-RoPE model (Qwen2-VL / Qwen2.5-VL / Qwen3-VL /
# Qwen3-Omni, ...) combined with speculative decoding when
# `num_speculative_tokens > 1` (the `== 1` case early-exits before the bug).
#
# We fix it here by aliasing `self.positions` to `self.mrope_positions` in
# the M-RoPE branch after __init__. `mrope_positions` has shape
# (3, max_num_tokens + 1), which makes the upstream slice
# `self.positions[:, last_token_indices]` produce the intended 3-D positions.
# ---------------------------------------------------------------------------
_orig_eagle_init = EagleProposer.__init__


def _patched_eagle_init(self, *args, **kwargs):
    _orig_eagle_init(self, *args, **kwargs)
    if getattr(self, "uses_mrope", False) and not hasattr(self, "positions"):
        # Alias, not copy: both names refer to the same buffer, so any
        # in-place update via `_set_positions` / `mrope_positions` is
        # visible through `positions` as well.
        self.positions = self.mrope_positions


EagleProposer.__init__ = _patched_eagle_init
