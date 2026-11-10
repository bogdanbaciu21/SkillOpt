"""Repeated validation samples for the nightly gate (``gate_rollouts``)."""

from __future__ import annotations

import importlib
import json
import math
import os
from collections import Counter

import pytest

from skillopt_sleep.backend import Backend, CliBackend, DualBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.consolidate import (
    GATE_ROLLOUTS_MAX,
    _exact_mean,
    _exact_mean_of_means,
    _mean_result,
    consolidate,
    gate_rollouts_count,
)
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.evidence import EvidenceLog, read_events
from skillopt_sleep.experiments.personas import researcher_persona
from skillopt_sleep.memory import set_learned
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.replay import replay_one
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

RULE = "Keep the final answer to the single expected phrase."
MEMORY_RULE = "Prefer the short common name."
SKILL = set_learned("# Skill\n", [])
MEMORY = set_learned("# Memory\n", [])


class SampledBackend(Backend):
    """A stochastic agent made deterministic: the answer to a task depends on
    which rules are in the documents and on the sample id.

    ``script`` maps ``(task id, state)`` to a pattern such as ``"1001"``; sample
    ``i`` is right when character ``i`` (cyclically) is ``1``. State is 0 with
    no rule, 1 with the skill rule, 2 with the skill and the memory rule. A
    task without an entry for state 2 keeps its state 1 behaviour. A pattern
    may also be a list of floats for partial-credit scores.
    """

    name = "sampled-stub"

    def __init__(self, script, *, tokens_per_call: int = 0, propose_memory: bool = False):
        self.script = script
        self.tokens_per_call = tokens_per_call
        self.propose_memory = propose_memory
        self.calls: list[tuple[str, int, int, str]] = []    # task, state, sample, phase
        self._tokens = 0

    def _state(self, skill: str, memory: str) -> int:
        if MEMORY_RULE in memory:
            return 2
        return 1 if RULE in skill else 0

    def attempt(self, task, skill, memory, sample_id: int = 0) -> str:
        state = self._state(skill, memory)
        self.calls.append((task.id, state, sample_id, self.evidence_phase))
        self._tokens += self.tokens_per_call
        while (task.id, state) not in self.script and state > 0:
            state -= 1
        pattern = self.script[(task.id, state)]
        value = pattern[sample_id % len(pattern)]
        return repr(float(value))

    def judge(self, task, response):
        score = float(response)
        hard = 1.0 if score >= 1.0 else 0.0
        return hard, score, "matched" if hard else "did not match"

    def reflect(self, failures, successes, skill, memory, *, edit_budget,
                evolve_skill, evolve_memory):
        if evolve_skill:
            return [EditRecord(target="skill", op="add", content=RULE, rationale="contrast")]
        if evolve_memory and self.propose_memory:
            return [EditRecord(target="memory", op="add", content=MEMORY_RULE, rationale="c")]
        return []

    def tokens_used(self) -> int:
        return self._tokens


def _task(task_id: str, split: str, **extra) -> TaskRecord:
    return TaskRecord(id=task_id, project="test", intent=f"question {task_id}",
                      split=split, **extra)


def _run(script, *, val=("v1",), metric="hard", **kwargs):
    backend = SampledBackend({("train", 0): "0", ("train", 1): "1", **script})
    tasks = [_task("train", "train")] + [_task(v, "val") for v in val]
    kwargs.setdefault("evolve_memory", False)
    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric=metric, **kwargs)
    return backend, result


def _val_calls(backend: SampledBackend, task_id: str = "v1"):
    return [(state, sample) for tid, state, sample, _phase in backend.calls if tid == task_id]


# ── the default is the gate that shipped before the knob existed ─────────────


def test_default_draws_one_sample_per_validation_replay() -> None:
    script = {("v1", 0): "0", ("v1", 1): "1"}
    backend, result = _run(script)
    explicit_backend, explicit = _run(script, gate_rollouts=1)

    assert DEFAULTS["gate_rollouts"] == 1
    assert backend.calls == explicit_backend.calls
    # baseline, the edit trial, and the fresh final replay: one sample each
    assert _val_calls(backend) == [(0, 0), (1, 0), (1, 0)]
    assert result.accepted is explicit.accepted is True
    assert result.gate_trials == explicit.gate_trials
    # the records are the ones written before the key existed
    for trial in result.gate_trials:
        assert set(trial) == {
            "target", "baseline_score", "candidate_score", "accepted",
            "blocked_by_regression", "task_deltas",
        }
    assert all("samples" not in row for row in result.holdout_detail)


def test_default_keeps_working_with_a_four_argument_replay_batch_double(monkeypatch) -> None:
    # Existing tests replace consolidate.replay_batch with this exact shape.
    module = importlib.import_module("skillopt_sleep.consolidate")
    scores = iter([[0.0], [0.0], [1.0], [1.0]])

    def fake_replay_batch(backend, tasks, skill, memory):
        batch = next(scores)
        return [(t, ReplayResult(id=t.id, hard=v, soft=v)) for t, v in zip(tasks, batch)]

    monkeypatch.setattr(module, "replay_batch", fake_replay_batch)
    backend = SampledBackend({})
    result = module.consolidate(
        backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
        gate_metric="hard", evolve_memory=False,
    )
    assert result.accepted is True
    assert backend.calls == []


# ── validation of the setting ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad", [0, -3, GATE_ROLLOUTS_MAX + 1, 2.7, math.nan, math.inf, "3", True, False, [2]],
)
def test_an_invalid_value_stops_before_any_model_call(bad) -> None:
    backend = SampledBackend({("train", 0): "0", ("v1", 0): "0"})
    with pytest.raises(ValueError, match="gate_rollouts must be a whole number"):
        consolidate(backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
                    gate_rollouts=bad)
    assert backend.calls == []


def test_valid_values() -> None:
    assert GATE_ROLLOUTS_MAX == 16
    assert gate_rollouts_count(None) == 1
    assert gate_rollouts_count(1) == 1
    assert gate_rollouts_count(4.0) == 4
    assert gate_rollouts_count(GATE_ROLLOUTS_MAX) == GATE_ROLLOUTS_MAX


# ── what repeated samples change ─────────────────────────────────────────────


def test_improvement_hidden_by_one_sample_is_accepted_with_repeats() -> None:
    # Without the rule the agent is right once in four; with it, always. Sample
    # 0 happens to be right both times, so a one-sample gate sees a tie.
    script = {("v1", 0): "1000", ("v1", 1): "1111"}

    _, single = _run(script)
    backend, repeated = _run(script, gate_rollouts=4)

    assert single.accepted is False
    assert single.baseline_score == single.candidate_score == 1.0
    assert repeated.accepted is True
    assert repeated.baseline_score == pytest.approx(0.25)
    assert repeated.candidate_score == pytest.approx(1.0)
    assert RULE in repeated.new_skill
    assert RULE not in single.new_skill
    assert Counter(_val_calls(backend)) == Counter(
        {(0, s): 1 for s in range(4)} | {(1, s): 2 for s in range(4)}
    )


def test_regression_hidden_by_one_lucky_sample_is_rejected_with_repeats() -> None:
    # The rule makes the agent worse (right three times in four -> once in
    # four), but sample 0 flips the other way.
    script = {("v1", 0): "0111", ("v1", 1): "1000"}

    _, single = _run(script)
    _, repeated = _run(script, gate_rollouts=4)

    assert single.accepted is True          # the one-sample gate is fooled
    assert repeated.accepted is False
    assert repeated.baseline_score == pytest.approx(0.75)
    assert repeated.gate_trials[0]["candidate_score"] == pytest.approx(0.25)
    assert RULE not in repeated.new_skill
    assert [edit.content for edit in repeated.rejected_edits] == [RULE]


@pytest.mark.parametrize("rollouts", [3, 5, 6, 7])
def test_equal_pass_counts_in_different_places_stay_a_tie(rollouts) -> None:
    # Same number of right answers overall, distributed differently across
    # tasks. With thirds or fifths a float mean of means can differ in the
    # last bit; a tie must stay a tie and be rejected.
    def pattern(passes: int) -> str:
        return "1" * passes + "0" * (rollouts - passes)

    base = (rollouts - 1, rollouts - 1, rollouts)
    cand = (rollouts, rollouts, rollouts - 2)
    assert sum(base) == sum(cand)
    script = {}
    for name, b, c in zip(("v1", "v2", "v3"), base, cand):
        script[(name, 0)] = pattern(b)
        script[(name, 1)] = pattern(c)

    _, result = _run(script, val=("v1", "v2", "v3"), gate_rollouts=rollouts)

    trial = result.gate_trials[0]
    assert trial["candidate_score"] == trial["baseline_score"]
    assert result.accepted is False
    assert result.gate_action == "reject"


def test_exact_mean_is_order_independent_and_fails_closed() -> None:
    thirds = [1.0, 1.0, 0.0]
    assert _exact_mean(thirds) == _exact_mean(list(reversed(thirds)))
    assert _exact_mean([0.1, 0.2, 0.3]) == _exact_mean([0.3, 0.1, 0.2])
    assert math.isnan(_exact_mean([1.0, math.nan]))
    assert math.isnan(_exact_mean([1.0, math.inf]))
    assert math.isnan(_exact_mean([]))


def test_mixed_metric_uses_means_of_both_scores() -> None:
    # Partial-credit scores: hard stays 0, soft moves 0.2 -> 0.6 on average.
    script = {("v1", 0): [0.2, 0.0, 0.4], ("v1", 1): [0.2, 0.9, 0.7]}

    _, single = _run(script, metric="mixed")
    _, repeated = _run(script, metric="mixed", gate_rollouts=3)

    assert single.accepted is False                       # sample 0 is 0.2 both times
    assert repeated.accepted is True
    assert repeated.baseline_score == pytest.approx(0.5 * 0.0 + 0.5 * 0.2)
    assert repeated.candidate_score == pytest.approx(0.5 * 0.0 + 0.5 * 0.6)


def test_no_regression_gate_compares_task_means() -> None:
    script = {
        ("v1", 0): "1000", ("v1", 1): "1111",   # improves on average
        ("v2", 0): "1111", ("v2", 1): "1110",   # regresses on average
    }

    _, single = _run(script, val=("v1", "v2"), gate_no_regression=True)
    _, repeated = _run(script, val=("v1", "v2"), gate_no_regression=True, gate_rollouts=4)

    # One sample: both tasks tie, so nothing regresses and nothing improves.
    assert single.accepted is False
    assert single.gate_trials[0]["blocked_by_regression"] is False
    # Four samples: the aggregate improves, and the per-task rule sees v2 fall.
    trial = repeated.gate_trials[0]
    assert trial["candidate_score"] > trial["baseline_score"]
    assert trial["blocked_by_regression"] is True
    assert repeated.accepted is False
    assert {row["task_id"]: row["status"] for row in trial["task_deltas"]} == {
        "v1": "improved", "v2": "regressed",
    }


def test_final_replay_applies_the_no_regression_rule_to_task_means() -> None:
    script = {
        ("v1", 0): "1000", ("v1", 1): "1111",
        ("v2", 0): "1111", ("v2", 1): "1110",
    }

    _, result = _run(script, val=("v1", "v2"), gate_no_regression=True, gate_rollouts=4)

    final = result.gate_trials[-1]
    assert final["target"] == "final"
    assert final["gate_rollouts"] == 4
    assert {row["task_id"]: row["status"] for row in final["task_deltas"]} == {
        "v1": "unchanged", "v2": "unchanged",       # the edit was not kept
    }
    assert final["accepted"] is False


def test_memory_candidates_are_sampled_the_same_way() -> None:
    script = {
        ("train", 0): "0", ("train", 1): "1",
        ("v1", 0): "1000", ("v1", 1): "1100", ("v1", 2): "1111",
    }
    backend = SampledBackend(script, propose_memory=True)
    tasks = [_task("train", "train"), _task("v1", "val")]

    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                         evolve_memory=True, gate_rollouts=4)

    skill_trial, memory_trial, final = result.gate_trials
    assert (skill_trial["target"], memory_trial["target"]) == ("skill", "memory")
    assert skill_trial["candidate_score"] == pytest.approx(0.5)
    assert memory_trial["baseline_score"] == pytest.approx(0.5)
    assert memory_trial["candidate_score"] == pytest.approx(1.0)
    assert memory_trial["gate_rollouts"] == 4 and memory_trial["accepted"] is True
    assert MEMORY_RULE in result.new_memory
    memory_phase = [(state, sample) for tid, state, sample, phase in backend.calls
                    if tid == "v1" and phase == "gate_trial:memory"]
    assert sorted(memory_phase) == [(2, s) for s in range(4)]


# ── scope ────────────────────────────────────────────────────────────────────


def test_training_replay_and_dream_rollouts_are_not_affected() -> None:
    script = {("v1", 0): "10", ("v1", 1): "11"}

    plain, _ = _run(script, rollouts_k=3)
    repeated, _ = _run(script, rollouts_k=3, gate_rollouts=4)

    def train_calls(backend):
        return [call[:3] for call in backend.calls if call[0] == "train"]

    assert train_calls(plain) == train_calls(repeated)
    assert {sample for _tid, _state, sample in train_calls(repeated)} == {0, 1, 2}


def test_gate_off_never_replays_validation(tmp_path) -> None:
    backend = SampledBackend({("train", 0): "0", ("train", 1): "1",
                              ("v1", 0): "0", ("v1", 1): "1"})
    path = os.path.join(tmp_path, "evidence.jsonl")
    backend.evidence = EvidenceLog(path)

    result = consolidate(
        backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
        gate_metric="hard", evolve_memory=False, gate_mode="off", gate_rollouts=5,
    )

    assert _val_calls(backend) == []
    assert result.gate_action == "greedy_applied"
    (baseline,) = [e for e in read_events(path)
                   if e.get("stage") == "gate" and e.get("event") == "baseline"]
    assert baseline["gate_rollouts"] == 0


def test_a_task_that_requires_a_tool_is_sampled_once() -> None:
    tool_task = TaskRecord(
        id="tool", project="test", intent="look it up", split="val",
        reference_kind="rule",
        judge={"checks": [{"op": "tool_called", "arg": "search"}]},
    )

    class ToolBackend(SampledBackend):
        def attempt_with_tools(self, task, skill, memory, tools):
            self.calls.append((task.id, self._state(skill, memory), 0, self.evidence_phase))
            return "TOOL_CALL: search", ["search"]

    backend = ToolBackend({("train", 0): "0", ("train", 1): "1",
                           ("v1", 0): "1000", ("v1", 1): "1111"})
    tasks = [_task("train", "train"), _task("v1", "val"), tool_task]

    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                         evolve_memory=False, gate_rollouts=4)

    baseline = [(tid, sample) for tid, _state, sample, phase in backend.calls
                if phase == "baseline_val"]
    assert Counter(tid for tid, _ in baseline) == {"v1": 4, "tool": 1}
    detail = {row["id"]: row for row in result.holdout_detail}
    assert (detail["v1"]["samples"], detail["tool"]["samples"]) == (4, 1)
    assert detail["v1"]["sample_hard"] == [1.0, 0.0, 0.0, 0.0]
    # each task counts once in the aggregate, whatever its number of samples
    assert result.baseline_score == pytest.approx((0.25 + 1.0) / 2)
    assert result.accepted is True


def test_a_night_without_a_disjoint_validation_slice_draws_one_sample() -> None:
    backend = SampledBackend({("train", 0): "0", ("train", 1): "1"})

    result = consolidate(backend, [_task("train", "train")], SKILL, MEMORY,
                         gate_metric="hard", evolve_memory=False, gate_rollouts=4)

    assert result.holdout_leaked is True
    assert result.gate_action == "reject_unverified"
    assert {sample for _tid, _state, sample, _phase in backend.calls} == {0}
    assert all(trial["gate_rollouts"] == 1 for trial in result.gate_trials)


def test_a_missing_task_result_still_fails_closed(monkeypatch) -> None:
    module = importlib.import_module("skillopt_sleep.consolidate")
    real = module.replay_batch

    def dropping_replay_batch(backend, tasks, skill, memory):
        pairs = real(backend, tasks, skill, memory)
        if RULE in skill:
            pairs = [pair for pair in pairs if pair[0].id != "v2"]
        return pairs

    monkeypatch.setattr(module, "replay_batch", dropping_replay_batch)
    script = {("v1", 0): "1000", ("v1", 1): "1111", ("v2", 0): "1111", ("v2", 1): "1111"}

    _, result = _run(script, val=("v1", "v2"), gate_no_regression=True, gate_rollouts=2)

    row = result.gate_trials[0]["task_deltas"][1]
    assert (row["task_id"], row["status"], row["candidate_score"]) == ("v2", "regressed", None)
    assert result.gate_trials[0]["blocked_by_regression"] is True
    assert result.accepted is False


def test_dream_consolidate_passes_the_setting_through() -> None:
    backend = SampledBackend({
        ("train", 0): "0", ("train", 1): "1",
        ("v1", 0): "1000", ("v1", 1): "1111",
    })
    tasks = [_task("train", "train"), _task("v1", "val")]

    result = dream_consolidate(
        backend, tasks, SKILL, MEMORY, gate_metric="hard", evolve_memory=False,
        gate_rollouts=4,
    )

    assert result.accepted is True
    assert {sample for _state, sample in _val_calls(backend)} == {0, 1, 2, 3}


def test_parallel_workers_give_the_same_decision_and_draws(monkeypatch) -> None:
    script = {("v1", 0): "1000", ("v1", 1): "1111", ("v2", 0): "0101", ("v2", 1): "0111"}
    serial_backend, serial = _run(script, val=("v1", "v2"), gate_rollouts=4)
    monkeypatch.setenv("SKILLOPT_SLEEP_WORKERS", "8")
    parallel_backend, parallel = _run(script, val=("v1", "v2"), gate_rollouts=4)

    assert parallel.gate_trials == serial.gate_trials
    assert parallel.accepted is serial.accepted is True
    assert Counter(c[:3] for c in parallel_backend.calls) == Counter(
        c[:3] for c in serial_backend.calls
    )


# ── records ──────────────────────────────────────────────────────────────────


def test_mean_result_reports_means_total_cost_and_a_failing_sample() -> None:
    samples = [
        ReplayResult(id="v", hard=1.0, soft=1.0, response="good", tokens=10,
                     latency_ms=5.0, tools_called=["search"]),
        ReplayResult(id="v", hard=0.0, soft=0.5, response="bad", tokens=20,
                     latency_ms=7.0, fail_reason="did not match",
                     tools_called=["search", "open"]),
        ReplayResult(id="v", hard=1.0, soft=1.0, response="good again", tokens=30,
                     latency_ms=9.0),
    ]

    mean = _mean_result(samples)

    assert mean.hard == pytest.approx(2 / 3)
    assert mean.soft == pytest.approx(2.5 / 3)
    assert mean.tokens == 60
    assert mean.latency_ms == pytest.approx(21.0)
    assert mean.response == "bad"
    assert mean.fail_reason == "did not match"
    assert mean.tools_called == ["search", "open"]
    assert samples[1].tools_called == ["search", "open"]     # inputs untouched
    assert samples[1].hard == 0.0


def test_trials_and_holdout_detail_record_the_samples() -> None:
    _, result = _run({("v1", 0): "1000", ("v1", 1): "1111"}, gate_rollouts=4)

    assert [trial["target"] for trial in result.gate_trials] == ["skill", "final"]
    assert all(trial["gate_rollouts"] == 4 for trial in result.gate_trials)
    (row,) = result.holdout_detail
    assert row["samples"] == 4
    assert row["sample_hard"] == [1.0, 0.0, 0.0, 0.0]
    assert row["hard"] == pytest.approx(0.25)
    # the detail shows a failing sample, not the lucky first one
    assert row["why"] == "did not match"
    json.dumps(result.gate_trials, allow_nan=False)
    json.dumps(result.holdout_detail, allow_nan=False)


def test_evidence_log_records_every_sample_and_the_setting(tmp_path) -> None:
    backend = SampledBackend({
        ("train", 0): "0", ("train", 1): "1",
        ("v1", 0): "1000", ("v1", 1): "1111",
    })
    path = os.path.join(tmp_path, "evidence.jsonl")
    backend.evidence = EvidenceLog(path)

    consolidate(
        backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
        gate_metric="hard", evolve_memory=False, gate_rollouts=4,
    )
    events = read_events(path)

    def of(kind, name):
        return [e for e in events if e.get("stage") == kind and e.get("event") == name]

    assert of("gate", "baseline")[0]["gate_rollouts"] == 4
    assert of("gate", "trial")[0]["gate_rollouts"] == 4
    baseline = [e for e in of("replay", "result") if e["phase"] == "baseline_val"]
    assert sorted(e["sample_id"] for e in baseline) == [0, 1, 2, 3]
    assert [e["hard"] for e in sorted(baseline, key=lambda e: e["sample_id"])] == [
        1.0, 0.0, 0.0, 0.0,
    ]


def test_every_sample_is_counted_by_the_backend_token_counter() -> None:
    script = {("v1", 0): "1000", ("v1", 1): "1111"}

    def spend(rollouts):
        backend = SampledBackend({("train", 0): "0", ("train", 1): "1", **script},
                                 tokens_per_call=7)
        consolidate(backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
                    gate_metric="hard", evolve_memory=False, gate_rollouts=rollouts)
        val_calls = len(_val_calls(backend))
        return backend.tokens_used(), val_calls, len(backend.calls) - val_calls

    single_tokens, single_val, single_other = spend(1)
    repeated_tokens, repeated_val, repeated_other = spend(4)

    assert repeated_val == 4 * single_val
    assert repeated_other == single_other          # nothing else the night did changed
    assert repeated_tokens - single_tokens == 7 * (repeated_val - single_val)


# ── budget ───────────────────────────────────────────────────────────────────


def test_budget_skip_of_the_final_replay_reuses_the_repeated_trial() -> None:
    script = {("v1", 0): "1000", ("v1", 1): "1111"}
    backend = SampledBackend({("train", 0): "0", ("train", 1): "1", **script})
    checks = iter([False, False, True, True])

    result = consolidate(
        backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
        gate_metric="hard", evolve_memory=True, gate_rollouts=4,
        budget_exhausted=lambda: next(checks, True),
    )

    assert result.budget_skipped == ["memory edits", "final validation replay"]
    assert result.accepted is True
    trial, final = result.gate_trials
    assert final["candidate_score"] == trial["candidate_score"] == pytest.approx(1.0)
    assert final["gate_rollouts"] == 4
    # baseline (4 samples) and one trial (4 samples); no final replay
    assert Counter(_val_calls(backend)) == Counter(
        {(0, s): 1 for s in range(4)} | {(1, s): 1 for s in range(4)}
    )


# ── caching backends ─────────────────────────────────────────────────────────


class _CountingCli(CliBackend):
    """A caching CLI backend with a scripted model call."""

    name = "counting-cli"

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.prompts.append(prompt)
        return f"answer {len(self.prompts)}"

    def judge(self, task, response):
        return 1.0, 1.0, ""


def test_a_caching_backend_resamples_each_sample_and_caches_a_repeat() -> None:
    backend = _CountingCli()
    task = _task("v1", "val")

    first = [replay_one(backend, task, "skill", "", sample_id) for sample_id in range(3)]
    calls_after_first = len(backend.prompts)
    again = [replay_one(backend, task, "skill", "", sample_id) for sample_id in range(3)]

    # three distinct draws, not one cached response reused three times
    assert calls_after_first == 3
    assert len({result.response for result in first}) == 3
    # an identical later replay (the gate's final replay) is served from cache
    assert len(backend.prompts) == calls_after_first
    assert [r.response for r in again] == [r.response for r in first]


# ── nightly cycle wiring ─────────────────────────────────────────────────────


def _cfg(tmp_path, **extra):
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    return load_config(
        invoked_project=str(project), projects="invoked", backend="mock",
        claude_home=str(home / ".claude"), auto_adopt=False, **extra,
    )


def _spy(monkeypatch):
    import skillopt_sleep.cycle as cycle_module

    seen: list[dict] = []
    real = cycle_module.dream_consolidate

    def spy(backend, tasks, skill, memory, **kwargs):
        seen.append(dict(kwargs))
        return real(backend, tasks, skill, memory, **kwargs)

    monkeypatch.setattr(cycle_module, "dream_consolidate", spy)
    return seen


def _baseline_samples(events):
    return {
        e["sample_id"] for e in events
        if e.get("stage") == "replay" and e.get("event") == "result"
        and e.get("phase") == "baseline_val"
    }


def test_cycle_forwards_the_configured_value_and_reports_it(tmp_path, monkeypatch) -> None:
    seen = _spy(monkeypatch)
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    outcome = run_sleep_cycle(_cfg(tmp_path, gate_rollouts=3), seed_tasks=tasks)

    assert [kwargs["gate_rollouts"] for kwargs in seen] == [3]
    with open(os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8") as fh:
        report = fh.read()
    assert "- validation samples per task: 3" in report
    events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
    start = next(e for e in events if e.get("stage") == "cycle" and e.get("event") == "start")
    assert start["config"]["gate_rollouts"] == 3
    assert _baseline_samples(events) == {0, 1, 2}
    assert all(trial["gate_rollouts"] == 3 for trial in outcome.report.gate_trials)


def test_cycle_default_is_one_sample_and_adds_no_report_line(tmp_path, monkeypatch) -> None:
    seen = _spy(monkeypatch)
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    outcome = run_sleep_cycle(_cfg(tmp_path), seed_tasks=tasks)

    assert [kwargs["gate_rollouts"] for kwargs in seen] == [1]
    with open(os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8") as fh:
        report = fh.read()
    assert "validation samples per task" not in report
    events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
    assert _baseline_samples(events) == {0}


def test_cycle_with_the_gate_off_adds_no_report_line(tmp_path, monkeypatch) -> None:
    _spy(monkeypatch)
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    outcome = run_sleep_cycle(
        _cfg(tmp_path, gate_rollouts=3, gate_mode="off"), seed_tasks=tasks,
    )

    with open(os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8") as fh:
        assert "validation samples per task" not in fh.read()


@pytest.mark.parametrize("bad", [0, "3", 2.5, GATE_ROLLOUTS_MAX + 1, True])
def test_cycle_rejects_an_invalid_value_before_any_model_call(tmp_path, monkeypatch, bad) -> None:
    seen = _spy(monkeypatch)
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    with pytest.raises(ValueError, match="gate_rollouts must be a whole number"):
        run_sleep_cycle(_cfg(tmp_path, gate_rollouts=bad), seed_tasks=tasks)
    assert seen == []


def test_cycle_reads_the_value_from_a_config_file(tmp_path, monkeypatch) -> None:
    seen = _spy(monkeypatch)
    home = tmp_path / "home"
    home.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gate_rollouts": 2}), encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: str(config_path))
    cfg = load_config(
        invoked_project=str(project), projects="invoked",
        backend="mock", claude_home=str(home / ".claude"), auto_adopt=False,
    )
    assert cfg.get("gate_rollouts") == 2
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    run_sleep_cycle(cfg, seed_tasks=tasks)

    assert [kwargs["gate_rollouts"] for kwargs in seen] == [2]


def test_default_writes_the_evidence_events_it_wrote_before(tmp_path) -> None:
    def gate_events(**kwargs):
        backend = SampledBackend({("train", 0): "0", ("train", 1): "1",
                                  ("v1", 0): "0", ("v1", 1): "1"})
        path = os.path.join(tmp_path, f"evidence-{len(kwargs)}.jsonl")
        backend.evidence = EvidenceLog(path)
        consolidate(backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
                    gate_metric="hard", evolve_memory=False, **kwargs)
        return [e for e in read_events(path) if e.get("stage") == "gate"]

    default, sampled = gate_events(), gate_events(gate_rollouts=3)

    assert default and all("gate_rollouts" not in event for event in default)
    assert sampled and all(event["gate_rollouts"] == 3 for event in sampled
                           if event.get("event") in {"baseline", "trial"})


def test_cycle_default_start_event_config_has_no_new_key(tmp_path, monkeypatch) -> None:
    _spy(monkeypatch)
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    outcome = run_sleep_cycle(_cfg(tmp_path), seed_tasks=tasks)

    events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
    (start,) = [e for e in events if e.get("stage") == "cycle" and e.get("event") == "start"]
    assert "gate_rollouts" not in start["config"]
    assert all("gate_rollouts" not in trial for trial in outcome.report.gate_trials)


def test_with_a_dual_backend_the_samples_are_drawn_on_the_target_only() -> None:
    script = {("train", 0): "0", ("train", 1): "1", ("v1", 0): "1000", ("v1", 1): "1111"}
    target, optimizer = SampledBackend(script), SampledBackend(script)
    tasks = [_task("train", "train"), _task("v1", "val")]

    result = consolidate(DualBackend(target, optimizer), tasks, SKILL, MEMORY,
                         gate_metric="hard", evolve_memory=False, gate_rollouts=4)

    assert result.accepted is True
    # four samples each for the baseline, the edit trial and the final replay
    assert _val_calls(target) == [(0, 0), (0, 1), (0, 2), (0, 3)] + [(1, i) for i in range(4)] * 2
    assert optimizer.calls == []


def test_repeated_samples_and_the_training_side_check_work_together() -> None:
    # Repeated samples reveal an improvement on the validation task that one
    # sample hides, and the same rule breaks every training task. The
    # validation gate would accept it; the training-side check then rejects it.
    script = {("v1", 0): "1000", ("v1", 1): "1111",
              **{(f"t{i}", 0): "1" for i in range(5)}, **{(f"t{i}", 1): "0" for i in range(5)}}
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]

    def run(**kwargs):
        backend = SampledBackend(script)
        return backend, consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                                    evolve_memory=False, **kwargs)

    _, sampled = run(gate_rollouts=4)
    backend, both = run(gate_rollouts=4, gate_train_check=True)

    assert sampled.accepted is True                       # validation alone accepts
    assert both.accepted is False
    trial = both.gate_trials[0]
    assert trial["gate_rollouts"] == 4
    assert trial["blocked_by_train_check"] is True
    # the check replays each training task once; repeated samples are for validation only
    check_calls = [(tid, sample) for tid, _state, sample, phase in backend.calls
                   if phase.startswith("gate_train_check")]
    assert sorted(check_calls) == [(f"t{i}", 0) for i in range(5)]


def test_a_non_finite_sample_score_never_accepts() -> None:
    assert math.isnan(_exact_mean_of_means([]))
    assert math.isnan(_exact_mean_of_means([[1.0, 1.0], [1.0, math.inf]]))
    assert _exact_mean_of_means([[1.0, 0.0], [1.0]]) == 0.75

    class OneBadSample(SampledBackend):
        def judge(self, task, response):
            hard, soft, why = super().judge(task, response)
            # one candidate sample of the validation task comes back without a usable score
            if task.id == "v1" and self.calls[-1][1:3] == (1, 2):
                return math.nan, math.nan, "no score"
            return hard, soft, why

    backend = OneBadSample({("train", 0): "0", ("train", 1): "1",
                            ("v1", 0): "0000", ("v1", 1): "1111"})
    result = consolidate(backend, [_task("train", "train"), _task("v1", "val")], SKILL, MEMORY,
                         gate_metric="hard", evolve_memory=False, gate_rollouts=4)

    assert result.accepted is False
