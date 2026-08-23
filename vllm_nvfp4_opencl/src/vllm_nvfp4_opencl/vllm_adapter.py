"""Narrow request lifecycle seam for a vLLM model-runner integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch

from .runtime import Profile
from .serving import Qwen35CadenceWeights, Qwen35PagedScheduler


class VllmCadenceAdapter:
    """Translate vLLM-style request batches into the resident scheduler.

    This intentionally stops at a four-layer hidden-state boundary. A model
    runner can call it between embedding and final-norm/lm-head work without
    exposing OpenCL handles to vLLM's scheduler.
    """

    def __init__(
        self,
        weights: Qwen35CadenceWeights,
        *,
        max_pages: int,
        default_max_tokens: int,
        max_batch_size: int = 4,
    ):
        if default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be positive")
        self.weights = weights
        self.scheduler = weights.create_paged_scheduler(
            max_pages, max_batch_size=max_batch_size
        )
        self.default_max_tokens = default_max_tokens
        self._closed = False

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(self.scheduler.sessions)

    def execute(
        self,
        request_ids: Sequence[str],
        hidden_states: torch.Tensor,
        *,
        new_max_tokens: Mapping[str, int] | None = None,
        finished_request_ids: Sequence[str] = (),
    ) -> tuple[torch.Tensor, Profile]:
        if self._closed:
            raise RuntimeError("vLLM cadence adapter is closed")
        if hidden_states.device.type != "cpu":
            raise ValueError("hidden_states must be a CPU tensor")
        if hidden_states.ndim != 2 or hidden_states.shape != (len(request_ids), 5120):
            raise ValueError("hidden_states must have shape [requests, 5120]")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within a decode step")

        active_in_batch = set(request_ids)
        for request_id in finished_request_ids:
            if request_id in active_in_batch:
                raise ValueError(
                    f"request cannot be both scheduled and finished: {request_id}"
                )
            if request_id in self.scheduler.sessions:
                self.scheduler.remove_request(request_id)

        capacities = new_max_tokens or {}
        for request_id in request_ids:
            if request_id not in self.scheduler.sessions:
                self.scheduler.add_request(
                    request_id,
                    capacities.get(request_id, self.default_max_tokens),
                )

        original_dtype = hidden_states.dtype
        host = hidden_states.detach().to(torch.float32).contiguous().numpy()
        outputs, profile = self.scheduler.decode_batch(
            {
                request_id: np.ascontiguousarray(host[index : index + 1])
                for index, request_id in enumerate(request_ids)
            }
        )
        result = np.concatenate(
            [outputs[request_id] for request_id in request_ids], axis=0
        )
        return torch.from_numpy(result).to(original_dtype), profile

    def abort(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            if request_id in self.scheduler.sessions:
                self.scheduler.remove_request(request_id)

    def execute_scheduler_output(
        self,
        scheduler_output: Any,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, Profile]:
        """Consume request-major token chunks from vLLM V1 SchedulerOutput.

        Multi-token chunks run one temporal slice at a time. This is a correct
        prefill baseline; projection GEMM batching remains a later optimization.
        Preemption and resume are rejected until state transfer exists.
        """
        scheduled = scheduler_output.num_scheduled_tokens
        if not scheduled or any(count <= 0 for count in scheduled.values()):
            raise ValueError("scheduler output must contain positive token counts")
        preempted = getattr(scheduler_output, "preempted_req_ids", None)
        resumed = getattr(
            getattr(scheduler_output, "scheduled_cached_reqs", None),
            "resumed_req_ids",
            set(),
        )
        if preempted or resumed:
            raise NotImplementedError(
                "resident cadence state transfer for preemption/resume is not implemented"
            )
        request_ids = list(scheduled)
        active = set(request_ids)
        for request_id in scheduler_output.finished_req_ids:
            if request_id in active:
                raise ValueError(
                    f"request cannot be both scheduled and finished: {request_id}"
                )
            if request_id in self.scheduler.sessions:
                self.scheduler.remove_request(request_id)
        for request_id in request_ids:
            if request_id not in self.scheduler.sessions:
                self.scheduler.add_request(request_id, self.default_max_tokens)

        total_tokens = sum(scheduled.values())
        if hidden_states.device.type != "cpu":
            raise ValueError("hidden_states must be a CPU tensor")
        if hidden_states.ndim != 2 or hidden_states.shape != (total_tokens, 5120):
            raise ValueError("hidden_states must have shape [scheduled tokens, 5120]")
        original_dtype = hidden_states.dtype
        host = hidden_states.detach().to(torch.float32).contiguous().numpy()
        slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for request_id, count in scheduled.items():
            slices[request_id] = (offset, offset + count)
            offset += count
        result = np.empty_like(host)
        aggregate = Profile()
        for token_index in range(max(scheduled.values())):
            temporal_batch = {
                request_id: np.ascontiguousarray(
                    host[slices[request_id][0] + token_index :
                         slices[request_id][0] + token_index + 1]
                )
                for request_id, count in scheduled.items()
                if token_index < count
            }
            outputs, profile = self.scheduler.decode_batch(temporal_batch)
            aggregate.upload_ns += profile.upload_ns
            aggregate.kernel_ns += profile.kernel_ns
            aggregate.download_ns += profile.download_ns
            for request_id, output in outputs.items():
                destination = slices[request_id][0] + token_index
                result[destination] = output[0]
        return torch.from_numpy(result).to(original_dtype), aggregate

    def close(self) -> None:
        if self._closed:
            return
        self.scheduler.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


__all__ = ["VllmCadenceAdapter"]
