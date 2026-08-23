"""Serialization and attribution helpers for native OpenCL command traces."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def _aggregate(events: list[Any], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for event in events:
        values = grouped[getattr(event, field)]
        values[0] += 1
        values[1] += event.duration_ns
    total_ns = sum(event.duration_ns for event in events)
    return [
        {
            field: label,
            "events": count,
            "kernel_ms": duration_ns / 1e6,
            "kernel_fraction": duration_ns / total_ns if total_ns else 0.0,
        }
        for label, (count, duration_ns) in sorted(
            grouped.items(), key=lambda item: item[1][1], reverse=True
        )
    ]


def _aggregate_stages(events: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for event in events:
        stage = event.scope.rsplit(".", 1)[-1]
        values = grouped[stage]
        values[0] += 1
        values[1] += event.duration_ns
    total_ns = sum(event.duration_ns for event in events)
    return [
        {
            "stage": stage,
            "events": count,
            "kernel_ms": duration_ns / 1e6,
            "kernel_fraction": duration_ns / total_ns if total_ns else 0.0,
        }
        for stage, (count, duration_ns) in sorted(
            grouped.items(), key=lambda item: item[1][1], reverse=True
        )
    ]


def _aggregate_stage_operations(events: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for event in events:
        stage = event.scope.rsplit(".", 1)[-1]
        values = grouped[(stage, event.operation)]
        values[0] += 1
        values[1] += event.duration_ns
    total_ns = sum(event.duration_ns for event in events)
    return [
        {
            "stage": stage,
            "operation": operation,
            "events": count,
            "kernel_ms": duration_ns / 1e6,
            "kernel_fraction": duration_ns / total_ns if total_ns else 0.0,
        }
        for (stage, operation), (count, duration_ns) in sorted(
            grouped.items(), key=lambda item: item[1][1], reverse=True
        )
    ]


def summarize_trace(events: list[Any]) -> dict[str, Any]:
    """Return a compact summary plus a normalized Chrome-like event stream."""
    if not events:
        raise ValueError("a completed trace must contain at least one event")
    origin_ns = min(event.queued_ns for event in events)
    kernel_ns = sum(event.duration_ns for event in events)
    first_start_ns = min(event.start_ns for event in events)
    final_end_ns = max(event.end_ns for event in events)
    device_span_ns = final_end_ns - first_start_ns
    return {
        "clock": "OpenCL device profiling clock",
        "event_count": len(events),
        "kernel_sum_ms": kernel_ns / 1e6,
        "device_span_ms": device_span_ns / 1e6,
        "inter_kernel_gap_ms": max(device_span_ns - kernel_ns, 0) / 1e6,
        "queued_to_complete_ms": (final_end_ns - origin_ns) / 1e6,
        "by_stage": _aggregate_stages(events),
        "by_stage_operation": _aggregate_stage_operations(events),
        "by_scope": _aggregate(events, "scope"),
        "by_operation": _aggregate(events, "operation"),
        "events": [
            {
                "sequence": index,
                "scope": event.scope,
                "operation": event.operation,
                "queued_us": (event.queued_ns - origin_ns) / 1e3,
                "submit_us": (event.submit_ns - origin_ns) / 1e3,
                "start_us": (event.start_ns - origin_ns) / 1e3,
                "end_us": (event.end_ns - origin_ns) / 1e3,
                "duration_us": event.duration_ns / 1e3,
            }
            for index, event in enumerate(events)
        ],
    }


def _sample_stats(values: list[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "samples": values,
    }


def _sample_table(
    traces: list[dict[str, Any]],
    field: str,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows_by_trace = [
        {tuple(row[key] for key in keys): row for row in trace[field]}
        for trace in traces
    ]
    expected = set(rows_by_trace[0])
    if any(set(rows) != expected for rows in rows_by_trace[1:]):
        raise ValueError(f"trace samples disagree on {field} labels")
    rows = []
    for identity in expected:
        times = [rows[identity]["kernel_ms"] for rows in rows_by_trace]
        fractions = [
            rows[identity]["kernel_fraction"] for rows in rows_by_trace
        ]
        row = dict(zip(keys, identity, strict=True))
        row.update(
            events=rows_by_trace[0][identity]["events"],
            kernel_ms=_sample_stats(times),
            kernel_fraction_median=statistics.median(fractions),
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row["kernel_ms"]["median"], reverse=True)


def summarize_trace_samples(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a median raw trace and attach cross-run timing distributions."""
    if not traces:
        raise ValueError("at least one trace sample is required")
    counts = {trace["event_count"] for trace in traces}
    if len(counts) != 1:
        raise ValueError(f"trace samples disagree on event count: {counts}")
    kernel_values = [trace["kernel_sum_ms"] for trace in traces]
    kernel_median = statistics.median(kernel_values)
    representative_index = min(
        range(len(traces)),
        key=lambda index: abs(kernel_values[index] - kernel_median),
    )
    result = dict(traces[representative_index])
    result["sampling"] = {
        "sample_count": len(traces),
        "representative_sample": representative_index,
        "kernel_sum_ms": _sample_stats(kernel_values),
        "device_span_ms": _sample_stats(
            [trace["device_span_ms"] for trace in traces]
        ),
        "inter_kernel_gap_ms": _sample_stats(
            [trace["inter_kernel_gap_ms"] for trace in traces]
        ),
        "replay_reported_kernel_ms": _sample_stats(
            [trace["replay_reported_kernel_ms"] for trace in traces]
        ),
        "replay_wall_ms": _sample_stats(
            [trace["replay_wall_ms"] for trace in traces]
        ),
        "by_stage": _sample_table(traces, "by_stage", ("stage",)),
        "by_stage_operation": _sample_table(
            traces, "by_stage_operation", ("stage", "operation")
        ),
        "by_operation": _sample_table(
            traces, "by_operation", ("operation",)
        ),
    }
    return result
