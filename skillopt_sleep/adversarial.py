"""Candidate-level robustness probes for SkillOpt-Sleep.

The nightly gate measures a candidate on held-out tasks, but a candidate can
still be brittle to harmless changes in how a request is framed.  This module
creates bounded, deterministic variants of *real training tasks* and compares
the candidate's score on each source task with its score on the corresponding
variant.

Probes are evidence, not training examples: they never enter validation/test
splits and they never influence reflection.  The caller decides whether a
flag is advisory or blocks a candidate.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence, Tuple

from skillopt_sleep.backend import Backend
from skillopt_sleep.gate import select_gate_score
from skillopt_sleep.replay import replay_batch
from skillopt_sleep.types import ReplayResult, TaskRecord

MAX_PROBES_PER_TASK = 3
MAX_ADVERSARIAL_PROBES = 256


def _normalize_split(value: str) -> str:
    return {"replay": "train", "holdout": "val"}.get(value, value)


def _strip_polite_frame(intent: str) -> str:
    """Remove only well-known request boilerplate; keep task semantics intact."""
    patterns = (
        r"(?is)^\s*please\s+",
        r"(?is)^\s*(?:can|could|would)\s+you\s+",
        r"(?is)^\s*i\s+(?:need|want)\s+you\s+to\s+",
    )
    for pattern in patterns:
        rewritten, count = re.subn(pattern, "", intent, count=1)
        if count and rewritten.strip():
            rewritten = rewritten.strip()
            return rewritten[:1].upper() + rewritten[1:]
    return ""


def _variant_intents(intent: str) -> List[Tuple[str, str]]:
    """Return conservative, deterministic surface variants in priority order."""
    raw = str(intent or "").strip()
    if not raw:
        return []
    variants: List[Tuple[str, str]] = []
    reframed = _strip_polite_frame(raw)
    if reframed and reframed != raw:
        variants.append(("request-frame", reframed))
    variants.extend((
        (
            "section-wrapper",
            f"Task to complete:\n\n{raw}\n\nRespond to the task above.",
        ),
        (
            "delimiter-wrapper",
            f"The request is between the markers.\n<request>\n{raw}\n</request>",
        ),
        (
            "boundary-shift",
            f"Use the following request as the complete instruction:\n---\n{raw}\n---",
        ),
    ))
    out: List[Tuple[str, str]] = []
    seen = {raw}
    for kind, value in variants:
        if value not in seen:
            seen.add(value)
            out.append((kind, value))
    return out


def generate_adversarial_probes(
    tasks: Sequence[TaskRecord],
    *,
    factor: int = 1,
) -> List[TaskRecord]:
    """Create bounded semantic-preserving probes from real TRAIN tasks only.

    ``factor`` is the maximum variants per source task and is capped at three.
    Synthetic, recalled, validation, and test records are excluded so probes
    cannot recycle held-out material or amplify already-derived tasks.
    """
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise ValueError("adversarial probe factor must be an integer")
    per_task = max(0, min(factor, MAX_PROBES_PER_TASK))
    if per_task == 0:
        return []
    out: List[TaskRecord] = []
    source_ids: set[str] = set()
    for task in tasks:
        if len(out) >= MAX_ADVERSARIAL_PROBES:
            break
        if _normalize_split(task.split) != "train" or task.origin != "real":
            continue
        if task.derived_from or "recall" in (task.tags or []):
            continue
        if task.id in source_ids:
            raise ValueError(
                f"adversarial probe source ids must be unique: {task.id!r}"
            )
        source_ids.add(task.id)
        for kind, intent in _variant_intents(task.intent)[:per_task]:
            if len(out) >= MAX_ADVERSARIAL_PROBES:
                break
            out.append(TaskRecord(
                id=f"{task.id}_adversarial_{kind}",
                project=task.project,
                intent=intent,
                context_excerpt=task.context_excerpt,
                system=task.system,
                attempted_solution=task.attempted_solution,
                outcome=task.outcome,
                reference_kind=task.reference_kind,
                reference=task.reference,
                judge=dict(task.judge),
                tags=list(task.tags) + ["dream", "adversarial", f"probe:{kind}"],
                source_sessions=list(task.source_sessions),
                split="train",
                origin="dream",
                derived_from=task.id,
                skill_hint=task.skill_hint,
            ))
    return out


def _score(result: ReplayResult, metric: str, mixed_weight: float) -> float | None:
    value = select_gate_score(result.hard, result.soft, metric, mixed_weight)
    return value if math.isfinite(value) else None


def evaluate_adversarial_probes(
    backend: Backend,
    tasks: Sequence[TaskRecord],
    skill: str,
    memory: str,
    *,
    factor: int = 1,
    metric: str = "mixed",
    mixed_weight: float = 0.5,
    margin: float = 0.0,
) -> Dict[str, Any]:
    """Score source/probe pairs and return a JSON-safe brittleness report.

    A row is flagged when its probe score falls more than ``margin`` below the
    matching source score. Non-finite scores are invalid and fail closed.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError("adversarial probe margin must be a finite number in [0, 1]")
    numeric_margin = float(margin)
    if not math.isfinite(numeric_margin) or not 0.0 <= numeric_margin <= 1.0:
        raise ValueError("adversarial probe margin must be a finite number in [0, 1]")

    probes = generate_adversarial_probes(tasks, factor=factor)
    source_by_id = {
        task.id: task
        for task in tasks
        if _normalize_split(task.split) == "train"
        and task.origin == "real"
        and not task.derived_from
        and "recall" not in (task.tags or [])
    }
    source_ids = list(dict.fromkeys(probe.derived_from for probe in probes))
    source_tasks = [source_by_id[source_id] for source_id in source_ids]
    source_pairs = replay_batch(backend, source_tasks, skill, memory)
    probe_pairs = replay_batch(backend, probes, skill, memory)
    source_results = {task.id: result for task, result in source_pairs}

    rows: List[Dict[str, Any]] = []
    flagged = 0
    invalid = 0
    deltas: List[float] = []
    for probe, probe_result in probe_pairs:
        source_result = source_results.get(probe.derived_from)
        source_score = (
            _score(source_result, metric, mixed_weight)
            if source_result is not None
            else None
        )
        probe_score = _score(probe_result, metric, mixed_weight)
        valid = source_score is not None and probe_score is not None
        delta = probe_score - source_score if valid else None
        is_flagged = not valid or bool(delta is not None and delta < -numeric_margin)
        if is_flagged:
            flagged += 1
        if not valid:
            invalid += 1
        if delta is not None:
            deltas.append(delta)
        kind = next(
            (tag.removeprefix("probe:") for tag in probe.tags if tag.startswith("probe:")),
            "unknown",
        )
        rows.append({
            "source_task_id": probe.derived_from,
            "probe_task_id": probe.id,
            "probe_kind": kind,
            "source_score": source_score,
            "probe_score": probe_score,
            "delta": delta,
            "status": "invalid" if not valid else ("brittle" if is_flagged else "stable"),
        })

    n = len(rows)
    return {
        "enabled": True,
        "factor": max(0, min(factor, MAX_PROBES_PER_TASK)),
        "margin": numeric_margin,
        "n_sources": len(source_tasks),
        "n_probes": n,
        "n_flagged": flagged,
        "n_invalid": invalid,
        "brittleness_rate": (flagged / n) if n else 0.0,
        "worst_delta": min(deltas) if deltas else None,
        "conclusive": n > 0,
        "flagged": flagged > 0,
        "rows": rows,
    }
