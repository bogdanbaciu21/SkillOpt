"""Opt-in live paired dream-off versus dream-on receipt with real evolution.

Every test here is SKIPPED unless ``SKILLOPT_TEST_REAL_LLM_DREAM=1`` is set,
because it makes real model calls. An opted-in run uses a real CLI backend for
BOTH the target and the optimizer:

* ``SKILLOPT_SLEEP_LIVE_DREAM_BACKEND=claude`` (default): the authenticated
  ``claude`` CLI. Dream generation additionally requires ``ANTHROPIC_API_KEY``
  because the ``--bare`` no-tools generation boundary only exists under
  API-key auth.
* ``SKILLOPT_SLEEP_LIVE_DREAM_BACKEND=opencode``: an installed OpenCode CLI
  with the user's login plus an explicit ``SKILLOPT_SLEEP_OPENCODE_MODEL``.
* ``SKILLOPT_SLEEP_LIVE_DREAM_BACKEND=azure_openai``: any OpenAI-compatible
  chat-completions server via ``AZURE_OPENAI_AUTH_MODE=openai_compatible``,
  ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``, and an explicit
  ``SKILLOPT_SLEEP_COMPAT_MODEL``.

The test runs ``dream_consolidate`` twice on the SAME task set with real
skill evolution enabled (``evolve_skill=True``): once with ``llm_dream``
off (template dreams) and once with it on. It writes a JSON receipt with,
per arm: baseline and candidate held-out scores, the held-out delta, applied
and rejected edit counts, the optimizer token delta, and, for the dream-on
arm, the generation acceptance/fallback accounting. The receipt path is
``SKILLOPT_LLM_DREAM_RECEIPT`` when set, else ``llm_dream_receipt.json`` in
the test's temporary directory, and the receipt is also printed to stdout.

Assertions are structural: both arms complete, holdout is never leaked,
evolution is really exercised (a reflect/gate decision happened), the
acceptance accounting is consistent, and the dream-on arm records a positive
generation token cost. The held-out deltas are REPORTED rather than asserted
because a live model may legitimately show no incremental lift on a given
scenario; the deterministic paired test in ``test_llm_dream.py``
(``TestPairedFunctionalEvidence``) pins the causal mechanism.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillopt_sleep.backend import ClaudeCliBackend, DualBackend
from skillopt_sleep.dream import (
    backend_fidelity_fn,
    backend_generate_fn,
    dream_consolidate,
)
from skillopt_sleep.types import TaskRecord

_LIVE_ENABLED = os.environ.get("SKILLOPT_TEST_REAL_LLM_DREAM", "").strip() == "1"

pytestmark = pytest.mark.skipif(
    not _LIVE_ENABLED,
    reason="set SKILLOPT_TEST_REAL_LLM_DREAM=1 to make real model calls",
)


def _build_backend():
    kind = (
        os.environ.get("SKILLOPT_SLEEP_LIVE_DREAM_BACKEND", "claude").strip()
        or "claude"
    )
    if kind == "claude":
        model = os.environ.get("SKILLOPT_SLEEP_CLAUDE_MODEL", "").strip() or "sonnet"
        return kind, model, lambda: ClaudeCliBackend(model=model)
    if kind == "opencode":
        from skillopt_sleep.backend import OpenCodeCliBackend

        model = os.environ.get("SKILLOPT_SLEEP_OPENCODE_MODEL", "").strip()
        if not model:
            pytest.fail(
                "SKILLOPT_SLEEP_OPENCODE_MODEL is required for opencode live runs",
                pytrace=False,
            )
        return kind, model, lambda: OpenCodeCliBackend(model=model)
    if kind == "azure_openai":
        from skillopt_sleep.backend import AzureOpenAIBackend

        model = os.environ.get("SKILLOPT_SLEEP_COMPAT_MODEL", "").strip()
        if not model:
            pytest.fail(
                "SKILLOPT_SLEEP_COMPAT_MODEL is required for azure_openai live "
                "runs (with AZURE_OPENAI_AUTH_MODE=openai_compatible, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_ENDPOINT set for any "
                "OpenAI-compatible server)",
                pytrace=False,
            )
        return kind, model, lambda: AzureOpenAIBackend(deployment=model)
    pytest.fail(
        f"unsupported SKILLOPT_SLEEP_LIVE_DREAM_BACKEND: {kind!r}", pytrace=False
    )


def _rule_task(tid: str, intent: str, topic: str, split: str) -> TaskRecord:
    return TaskRecord(
        id=tid,
        project="/live-dream",
        intent=intent,
        reference_kind="rule",
        reference="",
        judge={
            "checks": [
                {"op": "contains", "arg": "<answer>"},
                {"op": "contains", "arg": "</answer>"},
                {"op": "contains", "arg": topic},
            ]
        },
        split=split,
        origin="real",
        tags=["live-dream"],
    )


def _tasks() -> list[TaskRecord]:
    """A tiny convention-learning world: replies must wrap the answer in
    <answer></answer> tags, which a bare model does not do unprompted, so
    the baseline fails, reflection can learn the convention, and the gate
    measures the learned skill on differently phrased held-out tasks."""
    return [
        _rule_task(
            "live-train-sky",
            "State, in one word, the color of a cloudless midday sky.",
            "blue",
            "train",
        ),
        _rule_task(
            "live-train-planet",
            "Name, in one word, the planet humans live on.",
            "earth",
            "train",
        ),
        _rule_task(
            "live-val-capital",
            "What is the capital city of France? Answer briefly.",
            "paris",
            "val",
        ),
        _rule_task(
            "live-val-bees",
            "In one word, what do bees primarily produce?",
            "honey",
            "val",
        ),
    ]


class _Events:
    def __init__(self):
        self.rows = []

    def log(self, stage, event, **data):
        self.rows.append({"stage": stage, "event": event, **data})


def _require_clean_calls(name, *backends):
    """A receipt-grade run must not silently absorb provider failures: a dead
    key, an invalid model id, or an unreachable endpoint otherwise produces an
    all-zero receipt that looks like a model result."""
    for backend in backends:
        error = str(getattr(backend, "last_call_error", "") or "")
        if error:
            pytest.fail(
                f"{name} arm recorded a backend call error; the receipt is "
                f"invalid: {error[:300]}",
                pytrace=False,
            )


def _run_arm(build, *, llm_dream: bool):
    target = build()
    optimizer = build()
    backend = DualBackend(target, optimizer)
    events = _Events()
    tokens_before = backend.tokens_used()
    result = dream_consolidate(
        backend,
        _tasks(),
        skill="",
        memory="",
        dream_factor=1,
        llm_dream=llm_dream,
        generate_fn=backend_generate_fn(backend) if llm_dream else None,
        fidelity_fn=backend_fidelity_fn(backend) if llm_dream else None,
        gate_mode="on",
        evolve_skill=True,
        evolve_memory=False,
        evidence=events,
    )
    summaries = [r for r in events.rows if r.get("event") == "llm_dream_summary"]
    arm_backends = (target, optimizer)
    arm = {
        "baseline_holdout": result.holdout_baseline,
        "candidate_holdout": result.holdout_candidate,
        "holdout_delta": result.holdout_candidate - result.holdout_baseline,
        "gate_accepted_edit": bool(result.applied_edits),
        "applied_edits": len(result.applied_edits),
        "rejected_edits": len(result.rejected_edits),
        "unmatched_edits": len(result.unmatched_edits),
        "holdout_leaked": result.holdout_leaked,
        "backend_tokens_delta": backend.tokens_used() - tokens_before,
        "generation": summaries[0] if summaries else None,
        "new_skill_chars": len(result.new_skill),
        "gate_trials": len(result.gate_trials),
        "reflect_raw_empty": not result.reflect_raw,
    }
    return result, arm, arm_backends


def test_paired_dream_off_versus_dream_on_with_real_evolution(tmp_path):
    kind, model, build = _build_backend()

    off_result, off_arm, off_backends = _run_arm(build, llm_dream=False)
    on_result, on_arm, on_backends = _run_arm(build, llm_dream=True)

    receipt = {
        "backend": kind,
        "model": model,
        "tasks": [t.id for t in _tasks()],
        "dream_factor": 1,
        "evolve_skill": True,
        "evolve_memory": False,
        "arms": {"dream_off": off_arm, "dream_on": on_arm},
        "incremental_holdout_delta": on_arm["holdout_delta"]
        - off_arm["holdout_delta"],
    }
    receipt_path = Path(
        os.environ.get("SKILLOPT_LLM_DREAM_RECEIPT", "").strip()
        or tmp_path / "llm_dream_receipt.json"
    )
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("LLM_DREAM_LIVE_RECEIPT " + json.dumps(receipt))

    for name, arm_backends in (
        ("dream_off", off_backends),
        ("dream_on", on_backends),
    ):
        _require_clean_calls(name, *arm_backends)
    for name, result in (("dream_off", off_result), ("dream_on", on_result)):
        if result.holdout_leaked:
            pytest.fail(f"{name} arm leaked held-out data", pytrace=False)
        if not result.gate_trials and not result.applied_edits:
            pytest.fail(
                f"{name} arm never exercised evolution (no gate decision)",
                pytrace=False,
            )
    generation = on_arm["generation"]
    if not generation:
        pytest.fail("dream-on arm recorded no generation summary", pytrace=False)
    requested = generation.get("n_requested")
    accepted = generation.get("n_accepted")
    fallback = generation.get("n_fallback")
    if (
        not isinstance(requested, int)
        or not isinstance(accepted, int)
        or not isinstance(fallback, int)
        or requested < 1
        or accepted + fallback != requested
    ):
        pytest.fail("generation acceptance accounting is inconsistent", pytrace=False)
    if generation.get("optimizer_token_delta", 0) <= 0:
        pytest.fail("dream-on arm recorded no generation cost", pytrace=False)
    if off_arm["generation"] is not None:
        pytest.fail("dream-off arm must not run LLM generation", pytrace=False)
