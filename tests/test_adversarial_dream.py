"""Adversarial dream probes reject brittle candidate rules before adoption."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest

from skillopt_sleep.adversarial import (
    MAX_ADVERSARIAL_PROBES,
    evaluate_adversarial_probes,
    generate_adversarial_probes,
)
from skillopt_sleep.backend import DualBackend, MockBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.types import EditRecord, TaskRecord


def _task(
    task_id: str,
    *,
    intent: str = "Please return ok",
    split: str = "train",
    origin: str = "real",
    derived_from: str = "",
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        project="/project",
        intent=intent,
        context_excerpt="context",
        system="system",
        attempted_solution="prior",
        outcome="fail",
        reference_kind="exact",
        reference="ok",
        judge={"kind": "exact"},
        tags=["fixture"],
        source_sessions=["session-1"],
        split=split,
        origin=origin,
        derived_from=derived_from,
        skill_hint="demo-skill",
    )


class _CandidateBackend(MockBackend):
    """The planted rule works only on the harvested task's exact surface."""

    RULE = "Use the planted literal-only rule."
    HARVESTED_INTENTS = {"Please return ok", "Say ok"}

    def reflect(
        self,
        failures,
        successes,
        skill,
        memory,
        *,
        edit_budget,
        evolve_skill,
        evolve_memory,
    ):
        if self.RULE in f"{skill}\n{memory}":
            return []
        return [
            EditRecord(
                target="skill" if evolve_skill else "memory",
                op="add",
                content=self.RULE,
                rationale="planted brittle candidate",
            )
        ]

    def attempt(self, task, skill, memory, sample_id=0):
        if self.RULE not in f"{skill}\n{memory}":
            return "wrong"
        return task.reference if task.intent in self.HARVESTED_INTENTS else "wrong"


class _RobustCandidateBackend(_CandidateBackend):
    def attempt(self, task, skill, memory, sample_id=0):
        if self.RULE not in f"{skill}\n{memory}":
            return "wrong"
        return task.reference


class _CountingRobustBackend(_RobustCandidateBackend):
    def __init__(self) -> None:
        self.attempt_calls = 0

    def attempt(self, task, skill, memory, sample_id=0):
        self.attempt_calls += 1
        return super().attempt(task, skill, memory, sample_id)


class _NonFiniteProbeBackend(_RobustCandidateBackend):
    def judge(self, task, response):
        if task.origin == "dream":
            return float("nan"), 0.0, "invalid probe score"
        return super().judge(task, response)


class _CountingRoleBackend(_RobustCandidateBackend):
    def __init__(self, name: str) -> None:
        self.name = name
        self.attempt_calls = 0
        self.judge_calls = 0

    def attempt(self, task, skill, memory, sample_id=0):
        self.attempt_calls += 1
        return super().attempt(task, skill, memory, sample_id)

    def judge(self, task, response):
        self.judge_calls += 1
        return super().judge(task, response)


def _candidate_tasks() -> list[TaskRecord]:
    return [_task("train"), _task("val", intent="Say ok", split="val")]


def test_probe_generation_uses_only_real_underived_train_tasks() -> None:
    source = _task("source")
    tasks = [
        source,
        _task("val", split="val"),
        _task("test", split="test"),
        _task("dream", origin="dream", derived_from="source"),
        _task("recall", derived_from="old"),
    ]

    probes = generate_adversarial_probes(tasks, factor=3)

    assert len(probes) == 3
    assert {probe.derived_from for probe in probes} == {"source"}
    assert len({probe.id for probe in probes}) == 3
    assert all(probe.split == "train" and probe.origin == "dream" for probe in probes)
    assert all(probe.intent != source.intent for probe in probes)
    assert all("adversarial" in probe.tags for probe in probes)
    assert all(probe.reference == source.reference for probe in probes)
    assert all(probe.judge == source.judge for probe in probes)
    assert all(probe.system == source.system for probe in probes)
    assert all(probe.source_sessions == source.source_sessions for probe in probes)
    assert probes[0].intent == "Return ok"


def test_probe_generation_is_bounded_and_rejects_ambiguous_factor_types() -> None:
    tasks = [_task(f"task-{index}") for index in range(100)]

    probes = generate_adversarial_probes(tasks, factor=99)

    assert len(probes) == MAX_ADVERSARIAL_PROBES
    with pytest.raises(ValueError, match="must be an integer"):
        generate_adversarial_probes(tasks, factor=True)
    with pytest.raises(ValueError, match="source ids must be unique"):
        generate_adversarial_probes([_task("duplicate"), _task("duplicate")])


def test_probe_report_flags_a_planted_literal_surface_rule() -> None:
    report = evaluate_adversarial_probes(
        _CandidateBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        factor=2,
    )

    assert report["n_sources"] == 1
    assert report["n_probes"] == 2
    assert report["n_flagged"] == 2
    assert report["brittleness_rate"] == 1.0
    assert report["worst_delta"] == -1.0
    assert {row["status"] for row in report["rows"]} == {"brittle"}


def test_probe_report_keeps_a_surface_robust_rule_stable() -> None:
    report = evaluate_adversarial_probes(
        _RobustCandidateBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        factor=3,
    )

    assert report["n_flagged"] == 0
    assert report["flagged"] is False
    assert report["worst_delta"] == 0.0
    assert {row["status"] for row in report["rows"]} == {"stable"}


def test_dual_backend_probes_route_attempts_and_exact_judging_to_target() -> None:
    target = _CountingRoleBackend("target")
    optimizer = _CountingRoleBackend("optimizer")
    dual = DualBackend(target=target, optimizer=optimizer)

    report = evaluate_adversarial_probes(
        dual,
        [_task("train")],
        _CandidateBackend.RULE,
        "",
    )

    assert report["flagged"] is False
    assert target.attempt_calls == 2  # source + one probe
    assert target.judge_calls == 2
    assert optimizer.attempt_calls == 0
    assert optimizer.judge_calls == 0


def test_non_finite_probe_score_is_json_safe_and_fails_closed() -> None:
    report = evaluate_adversarial_probes(
        _NonFiniteProbeBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
    )

    assert report["flagged"] is True
    assert report["n_invalid"] == 1
    assert report["rows"][0]["probe_score"] is None
    assert report["rows"][0]["status"] == "invalid"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("margin", [True, -0.1, 1.1, float("nan"), "0.1"])
def test_probe_margin_rejects_non_finite_or_out_of_range_values(margin) -> None:
    with pytest.raises(ValueError, match="finite number"):
        evaluate_adversarial_probes(
            _RobustCandidateBackend(),
            [_task("train")],
            _CandidateBackend.RULE,
            "",
            margin=margin,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dream_adversarial": True}, "must be an integer"),
        ({"dream_adversarial": 1.5}, "must be an integer"),
        (
            {
                "dream_adversarial": 1,
                "dream_adversarial_blocking": "false",
            },
            "must be a boolean",
        ),
        (
            {"dream_adversarial": 1, "dream_adversarial_margin": float("inf")},
            "finite number",
        ),
    ],
)
def test_consolidate_rejects_ambiguous_adversarial_config(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        consolidate(
            _RobustCandidateBackend(),
            _candidate_tasks(),
            "# skill\n",
            "",
            evolve_memory=False,
            **overrides,
        )


def test_advisory_probe_surfaces_brittleness_without_changing_gate_decision() -> None:
    result = consolidate(
        _CandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=2,
        dream_adversarial_blocking=False,
    )

    assert result.accepted is True
    trial = result.gate_trials[0]
    assert trial["accepted"] is True
    assert trial["blocked_by_adversarial"] is False
    assert trial["adversarial_probe"]["flagged"] is True
    assert trial["adversarial_probe"]["blocking"] is False


def test_advisory_probe_is_held_out_result_equivalent_for_robust_candidate() -> None:
    off = consolidate(
        _RobustCandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
    )
    on = consolidate(
        _RobustCandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=3,
        dream_adversarial_blocking=False,
    )

    assert on.accepted == off.accepted is True
    assert on.gate_action == off.gate_action
    assert on.baseline_score == off.baseline_score
    assert on.candidate_score == off.candidate_score
    assert on.new_skill == off.new_skill
    assert on.new_memory == off.new_memory
    assert on.gate_trials[0]["adversarial_probe"]["n_flagged"] == 0


def test_blocking_probe_rejects_candidate_that_passed_the_held_out_gate() -> None:
    result = consolidate(
        _CandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=2,
        dream_adversarial_blocking=True,
    )

    assert result.accepted is False
    assert result.applied_edits == []
    assert [edit.content for edit in result.rejected_edits] == [_CandidateBackend.RULE]
    trial = result.gate_trials[0]
    assert trial["candidate_score"] == 1.0
    assert trial["accepted"] is False
    assert trial["blocked_by_adversarial"] is True
    assert trial["adversarial_probe"]["blocked"] is True


def test_blocking_probe_fails_closed_when_no_eligible_variant_exists() -> None:
    result = consolidate(
        _CandidateBackend(),
        [_task("train", intent=""), _task("val", intent="Say ok", split="val")],
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=1,
        dream_adversarial_blocking=True,
    )

    assert result.accepted is False
    probe = result.gate_trials[0]["adversarial_probe"]
    assert probe["conclusive"] is False
    assert probe["blocked"] is True
    assert probe["block_reason"] == "inconclusive_no_probes"


def test_default_off_is_result_identical_to_explicit_zero() -> None:
    default_backend = _CountingRobustBackend()
    explicit_backend = _CountingRobustBackend()
    default = consolidate(
        default_backend,
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
    )
    explicit = consolidate(
        explicit_backend,
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=0,
        dream_adversarial_blocking=False,
        dream_adversarial_margin=0.0,
    )

    assert asdict(default) == asdict(explicit)
    assert default_backend.attempt_calls == explicit_backend.attempt_calls == 4
    assert "adversarial_probe" not in default.gate_trials[0]
    assert "blocked_by_adversarial" not in default.gate_trials[0]
    assert DEFAULTS["dream_adversarial"] == 0
    assert DEFAULTS["dream_adversarial_blocking"] is False
    assert DEFAULTS["dream_adversarial_margin"] == 0.0
    assert load_config().get("dream_adversarial") == 0


def test_cycle_persists_advisory_probe_evidence_in_review_artifacts(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        evolve_memory=False,
        dream_adversarial=2,
        dream_adversarial_blocking=False,
        dream_adversarial_margin=0.015,
        auto_adopt=False,
    )

    tasks = _candidate_tasks()
    tasks[0].id = "train|<script>\n# heading"
    outcome = run_sleep_cycle(
        config,
        seed_tasks=tasks,
        backend=_CandidateBackend(),
    )

    with open(
        os.path.join(outcome.staging_dir, "diagnostics.json"),
        encoding="utf-8",
    ) as handle:
        diagnostics = json.load(handle)
    with open(
        os.path.join(outcome.staging_dir, "report.md"),
        encoding="utf-8",
    ) as handle:
        markdown = handle.read()
    assert diagnostics["dream_adversarial"] == 2
    assert diagnostics["dream_adversarial_blocking"] is False
    assert diagnostics["dream_adversarial_margin"] == 0.015
    assert diagnostics["gate_trials"] == outcome.report.gate_trials
    assert diagnostics["gate_trials"][0]["adversarial_probe"]["n_flagged"] == 2
    assert "adversarial dream probes: advisory" in markdown
    assert "Adversarial probes (advisory): 2 flagged / 2 total" in markdown
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
