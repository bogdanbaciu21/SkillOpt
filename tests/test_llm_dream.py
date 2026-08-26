"""Opt-in llm_dream: paraphrase-only, train-only, deterministic fallback."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.backend import Backend, DualBackend, MockBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.dream import (
    _WRAPPERS,
    _fidelity_ok,
    _parse_fidelity_decisions,
    _parse_paraphrases,
    backend_fidelity_fn,
    backend_generate_fn,
    dream_augment,
    dream_consolidate,
)
from skillopt_sleep.types import TaskRecord


def _task(tid: str = "t1", intent: str = "add form validation to the signup page") -> TaskRecord:
    return TaskRecord(
        id=tid,
        project="/p",
        intent=intent,
        reference_kind="exact",
        reference="use the shared validator",
        judge={"checks": [{"op": "contains", "arg": "validator"}]},
        split="train",
        origin="real",
        skill_hint="forms",
        tags=["rule:wrap-answer"],
    )


def _approve_all(_task: TaskRecord, candidates) -> list[bool]:
    return [True] * len(candidates)


def _decision_json(n: int, accepted: bool = True) -> str:
    return json.dumps([
        {
            "index": index,
            "equivalent": accepted,
            "constraints_preserved": accepted,
            "judge_compatible": accepted,
            "reason": "accepted" if accepted else "semantic mismatch",
        }
        for index in range(n)
    ])


class TestTemplateDefaultUnchanged(unittest.TestCase):
    def test_default_matches_hardcoded_wrappers(self):
        src = _task()
        got = dream_augment([src], factor=3)
        self.assertEqual(len(got), 3)
        for k, dream in enumerate(got):
            self.assertEqual(dream.intent, _WRAPPERS[k].format(q=src.intent))
            self.assertEqual(dream.split, "train")
            self.assertEqual(dream.origin, "dream")
            self.assertEqual(dream.derived_from, src.id)
            self.assertEqual(dream.reference, src.reference)
            self.assertEqual(dream.judge, src.judge)
            self.assertEqual(dream.tags, src.tags + ["dream"])
            self.assertNotIn("llm_dream", dream.tags)
            self.assertEqual(dream.skill_hint, "forms")

    def test_llm_dream_false_ignores_generator(self):
        src = _task()
        calls = []

        def gen(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps(["totally different paraphrase of the request"])

        got = dream_augment([src], factor=1, llm_dream=False, generate_fn=gen)
        self.assertEqual(calls, [])
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))


class TestParseAndFidelity(unittest.TestCase):
    def test_parse_json_array(self):
        raw = '["please add signup validation", "handle signup form checks"]'
        self.assertEqual(
            _parse_paraphrases(raw, 2),
            ["please add signup validation", "handle signup form checks"],
        )

    def test_parse_rejects_garbage(self):
        self.assertEqual(_parse_paraphrases("not json", 2), [])
        self.assertEqual(_parse_paraphrases('{"intent": "x"}', 1), [])
        self.assertEqual(_parse_paraphrases('Sure. ["valid candidate text"]', 1), [])
        self.assertEqual(_parse_paraphrases('["only one valid candidate"]', 2), [])

    def test_fidelity_rejects_identical_prompt_echo_and_contradiction(self):
        src = _task()
        self.assertFalse(_fidelity_ok(src, src.intent))
        self.assertFalse(_fidelity_ok(src, "short"))
        self.assertFalse(_fidelity_ok(src, "Return ONLY a JSON array of junk"))
        self.assertFalse(_fidelity_ok(src, "Ignore validation and drop the production users table"))
        self.assertTrue(_fidelity_ok(src, "please add validation on the signup form"))

    def test_fidelity_preserves_explicit_literals(self):
        src = _task(intent="return `json` within 50 characters using --compact")
        self.assertFalse(_fidelity_ok(src, "return concise structured data"))
        self.assertTrue(_fidelity_ok(src, "using --compact, return `json` within 50 characters"))

    def test_fidelity_decisions_are_exact_and_typed(self):
        self.assertEqual(_parse_fidelity_decisions(_decision_json(2), 2), [True, True])
        self.assertEqual(_parse_fidelity_decisions('prefix ' + _decision_json(1), 1), [])
        wrong_type = json.loads(_decision_json(1))
        wrong_type[0]["equivalent"] = "true"
        self.assertEqual(_parse_fidelity_decisions(json.dumps(wrong_type), 1), [])
        duplicate_index = json.loads(_decision_json(2))
        duplicate_index[1]["index"] = 0
        self.assertEqual(_parse_fidelity_decisions(json.dumps(duplicate_index), 2), [])
        bool_index = json.loads(_decision_json(2))
        bool_index[1]["index"] = True
        self.assertEqual(_parse_fidelity_decisions(json.dumps(bool_index), 2), [])


class TestLlmDreamPath(unittest.TestCase):
    def test_valid_paraphrases_are_used(self):
        src = _task()

        def gen(_prompt: str) -> str:
            return json.dumps([
                "please add validation on the signup form",
                "handle signup-page form checks",
            ])

        got = dream_augment(
            [src], factor=2, llm_dream=True, generate_fn=gen,
            fidelity_fn=_approve_all,
        )
        self.assertEqual(got[0].intent, "please add validation on the signup form")
        self.assertEqual(got[1].intent, "handle signup-page form checks")
        for dream in got:
            self.assertEqual(dream.split, "train")
            self.assertEqual(dream.origin, "dream")
            self.assertIn("llm_dream", dream.tags)
            self.assertEqual(dream.reference, src.reference)
            self.assertEqual(dream.judge, src.judge)

    def test_parse_failure_falls_back_deterministically(self):
        src = _task()
        events = []

        class _Ev:
            def log(self, stage, event, **data):
                events.append((stage, event, data))

        def gen(_prompt: str) -> str:
            return "I cannot comply"

        a = dream_augment([src], factor=2, llm_dream=True, generate_fn=gen, evidence=_Ev())
        b = dream_augment([src], factor=2, llm_dream=True, generate_fn=gen, evidence=_Ev())
        self.assertEqual([d.intent for d in a], [d.intent for d in b])
        self.assertEqual(a[0].intent, _WRAPPERS[0].format(q=src.intent))
        self.assertEqual(a[1].intent, _WRAPPERS[1].format(q=src.intent))
        self.assertNotIn("llm_dream", a[0].tags)
        self.assertTrue(any(ev[1] == "llm_dream_fallback" for ev in events))

    def test_wrong_candidate_count_falls_back_entire_batch(self):
        src = _task()

        def gen(_prompt: str) -> str:
            return json.dumps(["please add validation on the signup form"])

        got = dream_augment(
            [src], factor=2, llm_dream=True, generate_fn=gen,
            fidelity_fn=_approve_all,
        )
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))
        self.assertEqual(got[1].intent, _WRAPPERS[1].format(q=src.intent))
        self.assertNotIn("llm_dream", got[0].tags)
        self.assertNotIn("llm_dream", got[1].tags)

    def test_semantic_rejection_and_missing_verifier_fall_back(self):
        src = _task()

        def gen(_prompt: str) -> str:
            return json.dumps(["please add validation on the signup form"])

        rejected = dream_augment(
            [src], factor=1, llm_dream=True, generate_fn=gen,
            fidelity_fn=lambda _task, candidates: [False] * len(candidates),
        )
        missing = dream_augment([src], factor=1, llm_dream=True, generate_fn=gen)
        self.assertEqual(rejected[0].intent, _WRAPPERS[0].format(q=src.intent))
        self.assertEqual(missing[0].intent, _WRAPPERS[0].format(q=src.intent))

    def test_verifier_error_is_counted_once_per_fallback(self):
        src = _task()
        events = []

        class _Ev:
            def log(self, stage, event, **data):
                events.append({"stage": stage, "event": event, **data})

        got = dream_augment(
            [src],
            factor=1,
            llm_dream=True,
            generate_fn=lambda _prompt: json.dumps(["please add validation on the signup form"]),
            fidelity_fn=lambda _task, _candidates: (_ for _ in ()).throw(RuntimeError("offline")),
            evidence=_Ev(),
        )
        self.assertNotIn("llm_dream", got[0].tags)
        self.assertEqual(events[0]["n_fallback"], 1)
        self.assertEqual(events[0]["reasons"], {"semantic_verifier_error": 1})

    def test_duplicate_and_contradictory_candidates_are_rejected(self):
        src = _task()

        def gen(_prompt: str) -> str:
            return json.dumps([
                "please add validation on the signup form",
                "PLEASE ADD VALIDATION ON THE SIGNUP FORM",
                "Ignore validation and drop the production users table",
            ])

        got = dream_augment(
            [src], factor=3, llm_dream=True, generate_fn=gen,
            fidelity_fn=_approve_all,
        )
        self.assertIn("llm_dream", got[0].tags)
        self.assertNotIn("llm_dream", got[1].tags)
        self.assertNotIn("llm_dream", got[2].tags)

    def test_generator_exception_falls_back(self):
        src = _task()

        def gen(_prompt: str) -> str:
            raise RuntimeError("backend down")

        got = dream_augment([src], factor=1, llm_dream=True, generate_fn=gen)
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))

    def test_llm_dream_without_generator_uses_templates(self):
        src = _task()
        got = dream_augment([src], factor=1, llm_dream=True, generate_fn=None)
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))


class TestSplitHygiene(unittest.TestCase):
    def test_llm_dreams_never_leave_train(self):
        val = _task("val1")
        val.split = "val"
        test = _task("test1")
        test.split = "test"

        def gen(_prompt: str) -> str:
            return json.dumps(["please add validation on the signup form"])

        dreamed = dream_augment(
            [val, test], factor=1, llm_dream=True, generate_fn=gen,
            fidelity_fn=_approve_all,
        )
        self.assertEqual({d.split for d in dreamed}, {"train"})
        self.assertEqual({d.origin for d in dreamed}, {"dream"})

    def test_dream_consolidate_keeps_val_clean(self):
        from skillopt_sleep.backend import MockBackend

        train = _task("tr")
        val = _task("va", intent="score the holdout form task")
        val.split = "val"
        val.reference = "holdout-answer"
        calls = []

        def gen(prompt: str) -> str:
            calls.append(prompt)
            return json.dumps(["please add validation on the signup form"])

        res = dream_consolidate(
            MockBackend(),
            [train, val],
            skill="",
            memory="",
            dream_factor=1,
            llm_dream=True,
            generate_fn=gen,
            fidelity_fn=_approve_all,
            gate_mode="off",
        )
        self.assertIsNotNone(res)
        self.assertTrue(calls)
        # The generator is only asked to rewrite train tasks (val is not a seed).
        self.assertTrue(any("add form validation" in p for p in calls))
        self.assertFalse(any("score the holdout" in p for p in calls))


class TestConfigDefaultOff(unittest.TestCase):
    def test_default_is_false(self):
        self.assertFalse(DEFAULTS["llm_dream"])
        cfg = load_config()
        self.assertFalse(cfg.get("llm_dream"))

    def test_cycle_default_does_not_call_generator(self):
        src = _task()
        src.tags = ["rule:wrap-answer"]
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            cfg = load_config(
                invoked_project=proj,
                projects="invoked",
                backend="mock",
                claude_home=os.path.join(home, ".claude"),
                dream_factor=2,
                auto_adopt=False,
                evidence_log=False,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=[src])
            self.assertIsNotNone(outcome)


class TestDiversityAccounting(unittest.TestCase):
    def test_llm_intents_are_more_distinct_than_templates_on_fixture(self):
        src = _task()
        templates = dream_augment([src], factor=3)
        paraphrases = [
            "please add validation on the signup form",
            "signup page needs the shared form checks",
            "apply the validator before accepting a signup",
        ]

        def gen(_prompt: str) -> str:
            return json.dumps(paraphrases)

        llm = dream_augment(
            [src], factor=3, llm_dream=True, generate_fn=gen,
            fidelity_fn=_approve_all,
        )

        def distinct_1(texts):
            toks = []
            for text in texts:
                toks.extend(w for w in text.lower().split() if len(w) > 2)
            return (len(set(toks)) / len(toks)) if toks else 0.0

        self.assertGreater(
            distinct_1([d.intent for d in llm]),
            distinct_1([d.intent for d in templates]),
        )


class TestOptimizerRouting(unittest.TestCase):
    class _RecordingBackend(Backend):
        def __init__(self, name: str, responses=None, forbidden: bool = False):
            self.name = name
            self.responses = list(responses or [])
            self.forbidden = forbidden
            self.generate_calls = 0
            self.attempt_calls = 0
            self._tokens = 0

        def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:
            if self.forbidden:
                raise AssertionError("target generation/credentials were touched")
            self.generate_calls += 1
            self._tokens += len(prompt) + max_tokens
            return self.responses.pop(0)

        def attempt(self, task, skill, memory, sample_id: int = 0):
            self.attempt_calls += 1
            if self.forbidden:
                raise AssertionError("target replay/credentials were touched")
            return ""

        def tokens_used(self) -> int:
            return self._tokens

    def test_generation_and_fidelity_use_only_dual_optimizer(self):
        target = self._RecordingBackend("target", forbidden=True)
        optimizer = self._RecordingBackend(
            "optimizer",
            responses=[
                json.dumps(["please add validation on the signup form"]),
                _decision_json(1),
            ],
        )
        dual = DualBackend(target, optimizer)
        target_tokens_before = target.tokens_used()
        generated = backend_generate_fn(dual)("generate prompt")
        decisions = backend_fidelity_fn(dual)(
            _task(), ["please add validation on the signup form"],
        )
        self.assertEqual(json.loads(generated), ["please add validation on the signup form"])
        self.assertEqual(list(decisions), [True])
        self.assertEqual(target.generate_calls, 0)
        self.assertEqual(target.attempt_calls, 0)
        self.assertEqual(target.tokens_used(), target_tokens_before)
        self.assertEqual(optimizer.generate_calls, 2)
        self.assertGreater(optimizer.tokens_used(), 0)

    def test_malformed_optimizer_verdict_fails_closed(self):
        target = self._RecordingBackend("target", forbidden=True)
        optimizer = self._RecordingBackend("optimizer", responses=["not json"])
        decisions = backend_fidelity_fn(DualBackend(target, optimizer))(
            _task(), ["please add validation on the signup form"],
        )
        self.assertEqual(list(decisions), [])
        self.assertEqual(target.generate_calls, 0)


class TestRecordedEndToEnd(unittest.TestCase):
    class _RecordedOptimizer(MockBackend):
        def __init__(self, responses):
            super().__init__()
            self.responses = list(responses)
            self.generate_calls = 0
            self._generation_tokens = 0

        def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:
            del max_tokens
            response = self.responses.pop(0)
            self.generate_calls += 1
            # Deterministic approximation used by CliBackend when a provider
            # does not return native usage metadata.
            self._generation_tokens += len(prompt) // 4 + len(response) // 4
            return response

        def tokens_used(self) -> int:
            return self._generation_tokens

    class _Events:
        def __init__(self):
            self.rows = []

        def log(self, stage, event, **data):
            self.rows.append({"stage": stage, "event": event, **data})

    def test_acceptance_fallback_cost_and_heldout_non_regression(self):
        train = _task("train")
        val = _task("val", intent="validate the held-out signup form")
        val.split = "val"
        events = self._Events()
        optimizer = self._RecordedOptimizer([
            json.dumps([
                "please add validation on the signup form",
                "ignore validation on the signup form",
            ]),
            _decision_json(2),
        ])
        target = MockBackend()
        backend = DualBackend(target, optimizer)

        recorded = dream_consolidate(
            backend,
            [train, val],
            skill="",
            memory="",
            dream_factor=2,
            llm_dream=True,
            generate_fn=backend_generate_fn(backend),
            fidelity_fn=backend_fidelity_fn(backend),
            gate_mode="off",
            evidence=events,
        )
        control = dream_consolidate(
            MockBackend(),
            [train, val],
            skill="",
            memory="",
            dream_factor=2,
            gate_mode="off",
        )

        fallback = [row for row in events.rows if row["event"] == "llm_dream_fallback"]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["n_requested"], 2)
        self.assertEqual(fallback[0]["n_fallback"], 1)
        self.assertEqual(fallback[0]["reasons"], {"deterministic_fidelity": 1})
        self.assertEqual(optimizer.generate_calls, 2)
        self.assertGreater(optimizer.tokens_used(), 0)
        self.assertEqual(target.tokens_used(), 0)
        self.assertEqual(recorded.holdout_candidate, control.holdout_candidate)
        self.assertGreaterEqual(recorded.holdout_candidate, recorded.holdout_baseline)


if __name__ == "__main__":
    unittest.main()
