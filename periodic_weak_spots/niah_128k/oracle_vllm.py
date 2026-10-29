from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from flashinfer.norm import gemma_rmsnorm

from vllm.models.qwen4_exp.nvidia import model as qwen_model_module
from vllm.models.qwen4_exp.nvidia import qsa as qsa_module
from vllm.models.qwen4_exp.nvidia.indexer_qsa import (
    QSAIndexer,
    apply_qsa_rope,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention


@dataclass
class _OracleKeyState:
    keys: torch.Tensor
    seen: torch.Tensor
    last_position: int


@contextmanager
def _replace_global(module: ModuleType, name: str, value: Any) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def select_oracle_blocks(
    query: torch.Tensor,
    normalized_keys: torch.Tensor,
    *,
    visible_blocks: int,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    if query.ndim != 2:
        raise ValueError("query 必须具有 [index_heads, index_head_dim] 形状")
    if normalized_keys.ndim != 2:
        raise ValueError("normalized_keys 必须具有 [tokens, index_head_dim] 形状")
    if query.shape[-1] != normalized_keys.shape[-1]:
        raise ValueError("query 与 normalized_keys 的 head_dim 不一致")
    if visible_blocks < 0:
        raise ValueError("visible_blocks 不能为负数")
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("compress_ratio 与 token_topk 必须满足正数整除关系")
    complete_blocks = min(visible_blocks, normalized_keys.shape[0] // compress_ratio)
    block_topk = token_topk // compress_ratio
    selected_count = min(complete_blocks, block_topk)
    if selected_count == 0:
        return torch.empty(0, dtype=torch.int32, device=query.device)
    complete_tokens = complete_blocks * compress_ratio
    keys = normalized_keys[:complete_tokens].to(device=query.device)
    scores = torch.matmul(query.float(), keys.float().transpose(0, 1))
    token_scores = torch.relu(scores).sum(dim=0)
    block_scores = token_scores.reshape(complete_blocks, compress_ratio).sum(dim=1)
    return torch.topk(
        block_scores,
        selected_count,
        dim=0,
        largest=True,
        sorted=True,
    ).indices.to(torch.int32)


class OracleQSAIndexer(QSAIndexer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.index_kv_heads != 1:
            raise NotImplementedError("oracle indexer 当前要求 indexer_kv_heads=1")
        self._oracle_key_store: dict[int, _OracleKeyState] = {}

    def _normalize_keys(
        self,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = raw_keys.shape[0]
        keys = raw_keys.reshape(num_tokens, self.index_kv_heads, self.index_head_dim)
        keys = gemma_rmsnorm(
            keys.reshape(-1, self.index_head_dim),
            self.k_layernorm.weight,
            self.k_layernorm.variance_epsilon,
        ).reshape_as(keys)
        keys = apply_qsa_rope(self.rotary_emb, positions.clamp_min(0), keys)
        return keys[:, 0].to(torch.bfloat16)

    def _normalize_queries(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        projected_q, _ = projected_qk[:num_tokens].split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        queries = projected_q.reshape(num_tokens, self.index_n_heads, self.index_head_dim)
        queries = gemma_rmsnorm(
            queries.reshape(-1, self.index_head_dim),
            self.q_layernorm.weight,
            self.q_layernorm.variance_epsilon,
        ).reshape_as(queries)
        queries = apply_qsa_rope(
            self.rotary_emb,
            positions[..., :num_tokens].clamp_min(0),
            queries,
        )
        return queries.to(torch.bfloat16)

    def _state_for_request(self, request: int, required_tokens: int) -> _OracleKeyState:
        if required_tokens <= 0:
            raise ValueError("oracle indexer 的请求长度必须为正数")
        state = self._oracle_key_store.get(request)
        if state is not None and state.keys.shape[0] >= required_tokens:
            return state
        capacity = (
            required_tokens
            if state is None
            else max(required_tokens, state.keys.shape[0] + 4096)
        )
        keys = torch.empty(
            capacity,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
        seen = torch.zeros(capacity, dtype=torch.bool, device="cpu")
        last_position = -1
        if state is not None:
            keys[: state.keys.shape[0]].copy_(state.keys)
            seen[: state.seen.shape[0]].copy_(state.seen)
            last_position = state.last_position
        state = _OracleKeyState(keys=keys, seen=seen, last_position=last_position)
        self._oracle_key_store[request] = state
        return state

    def _store_keys(
        self,
        raw_metadata: Any,
        normalized_keys: torch.Tensor,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        requests = raw_metadata.token_to_req[:num_tokens]
        positions = raw_metadata.logical_positions[:num_tokens]
        sequence_lengths = raw_metadata.seq_lens
        valid_requests = (
            (requests >= 0)
            & (requests < sequence_lengths.numel())
            & (positions >= 0)
        )
        active_requests = torch.unique(requests[valid_requests]).tolist()
        for request_value in active_requests:
            request = int(request_value)
            row_mask = (requests == request) & valid_requests
            row_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
            if row_indices.numel() == 0:
                continue
            request_positions = positions.index_select(0, row_indices).long()
            minimum_position = int(request_positions.min().item())
            existing = self._oracle_key_store.get(request)
            if existing is not None and minimum_position <= existing.last_position:
                del self._oracle_key_store[request]
            maximum_position = int(request_positions.max().item())
            sequence_length = int(sequence_lengths[request].item())
            state = self._state_for_request(
                request,
                max(maximum_position + 1, sequence_length),
            )
            values = normalized_keys.index_select(0, row_indices).detach().to("cpu")
            positions_cpu = request_positions.to("cpu")
            state.keys.index_copy_(0, positions_cpu, values)
            state.seen.index_fill_(0, positions_cpu, True)
            state.last_position = max(state.last_position, maximum_position)

    def _write_oracle_rows(
        self,
        out: torch.Tensor,
        queries: torch.Tensor,
        raw_metadata: Any,
        compressed_metadata: Any,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        requests = raw_metadata.token_to_req[:num_tokens]
        positions = raw_metadata.logical_positions[:num_tokens]
        sequence_lengths = raw_metadata.seq_lens
        visible_blocks = compressed_metadata.visible_blocks[:num_tokens]
        if sequence_lengths.numel() == 0:
            raise RuntimeError("oracle indexer 收到的请求长度为空")
        valid_requests = (
            (requests >= 0)
            & (requests < sequence_lengths.numel())
            & (positions >= 0)
        )
        safe_requests = requests.clamp(0, sequence_lengths.numel() - 1).long()
        final_rows = valid_requests & (
            positions == sequence_lengths.index_select(0, safe_requests) - 1
        )
        final_row_indices = torch.nonzero(final_rows, as_tuple=False).flatten().tolist()
        output_width = self.output_width
        block_topk = self.token_topk // self.compress_ratio
        for row_value in final_row_indices:
            row = int(row_value)
            request = int(requests[row].item())
            position = int(positions[row].item())
            visible = int(visible_blocks[row].item())
            complete_blocks = min(visible, block_topk)
            complete_tokens = visible * self.compress_ratio
            state = self._oracle_key_store.get(request)
            if complete_tokens > 0:
                if state is None or state.keys.shape[0] < complete_tokens:
                    raise RuntimeError(
                        f"oracle indexer 缺少请求 {request} 的完整 key，position={position}"
                    )
                if not bool(state.seen[:complete_tokens].all().item()):
                    raise RuntimeError(
                        f"oracle indexer 的请求 {request} 存在未写入的历史 key，"
                        f"position={position}"
                    )
                selected_blocks = select_oracle_blocks(
                    queries[row],
                    state.keys,
                    visible_blocks=visible,
                    compress_ratio=self.compress_ratio,
                    token_topk=self.token_topk,
                )
            else:
                selected_blocks = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=out.device,
                )
            out[row].fill_(-1)
            expanded_count = complete_blocks * self.compress_ratio
            if expanded_count:
                offsets = torch.arange(
                    self.compress_ratio,
                    dtype=torch.int32,
                    device=out.device,
                )
                expanded = (
                    selected_blocks[:, None] * self.compress_ratio + offsets[None, :]
                ).reshape(-1)
                out[row, :expanded_count].copy_(expanded)
            tail_start = ((position + 1) // self.compress_ratio) * self.compress_ratio
            tail_count = (position + 1) - tail_start
            if tail_count:
                tail = torch.arange(
                    tail_start,
                    tail_start + tail_count,
                    dtype=torch.int32,
                    device=out.device,
                )
                out[row, expanded_count : expanded_count + tail_count].copy_(tail)
            out[row, output_width] = expanded_count + tail_count

    def forward(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        attn: Any,
        qkv: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        metadata = self._metadata()
        if metadata is None:
            return super().forward(
                projected_qk,
                positions,
                out,
                attn=attn,
                qkv=qkv,
                slot_mapping=slot_mapping,
            )
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        projected_qk = projected_qk[:num_tokens]
        positions = positions[..., :num_tokens]
        _, raw_keys = projected_qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        normalized_keys = self._normalize_keys(raw_keys, positions)
        self._store_keys(raw_metadata, normalized_keys)
        selected, main_outputs = super().forward(
            projected_qk,
            positions,
            out,
            attn=attn,
            qkv=qkv,
            slot_mapping=slot_mapping,
        )
        if selected.shape[0] != num_tokens:
            raise RuntimeError("oracle indexer 收到的 selection 行数不正确")
        queries = self._normalize_queries(projected_qk, positions, num_tokens)
        self._write_oracle_rows(selected, queries, raw_metadata, compressed_metadata)
        return selected, main_outputs


class OracleQwen4ExpQSAAttention(Qwen4ExpQSAAttention):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _replace_global(qsa_module, "QSAIndexer", OracleQSAIndexer):
            super().__init__(*args, **kwargs)


class OracleQwen4ExpForCausalLM(Qwen4ExpForCausalLM):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _replace_global(
            qwen_model_module,
            "Qwen4ExpQSAAttention",
            OracleQwen4ExpQSAAttention,
        ):
            super().__init__(*args, **kwargs)


class OracleQwen4ExpForConditionalGeneration(Qwen4ExpForConditionalGeneration):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _replace_global(
            qwen_model_module,
            "Qwen4ExpForCausalLM",
            OracleQwen4ExpForCausalLM,
        ):
            super().__init__(*args, **kwargs)


__all__ = [
    "OracleQSAIndexer",
    "OracleQwen4ExpForCausalLM",
    "OracleQwen4ExpForConditionalGeneration",
    "OracleQwen4ExpQSAAttention",
    "select_oracle_blocks",
]
