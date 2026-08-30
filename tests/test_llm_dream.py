"""Opt-in llm_dream: paraphrase-only, train-only, deterministic fallback."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.backend import (
    Backend,
    ClaudeCliBackend,
    CliBackend,
    DualBackend,
    MockBackend,
)
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.dream import (
    _WRAPPERS,
    _dedupe_text,
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

    def test_parse_accepts_single_whole_message_fence_only(self):
        fenced = '```json\n["please add signup validation"]\n```'
        self.assertEqual(
            _parse_paraphrases(fenced, 1), ["please add signup validation"]
        )
        self.assertEqual(_parse_paraphrases('```\n["ok candidate text"]\n```', 1),
                         ["ok candidate text"])
        self.assertEqual(_parse_paraphrases('Sure. ```json\n["x y z candidate"]\n```', 1), [])
        self.assertEqual(_parse_paraphrases('```json\n["x y z candidate"]\n``` done', 1), [])
        self.assertEqual(_parse_paraphrases('```json\nnot json\n```', 1), [])
        self.assertEqual(_parse_paraphrases('`["inline pseudo json"]`', 1), [])

    def test_fenced_fidelity_verdict_parses_and_stays_strict(self):
        verdict = _decision_json(1)
        self.assertEqual(
            _parse_fidelity_decisions(f"```json\n{verdict}\n```", 1), [True]
        )
        self.assertEqual(
            _parse_fidelity_decisions(f"ok ```json\n{verdict}\n```", 1), []
        )
        self.assertEqual(_parse_fidelity_decisions("```json\n{}\n```", 1), [])

    def test_parse_rejects_garbage(self):
        self.assertEqual(_parse_paraphrases("not json", 2), [])
        self.assertEqual(_parse_paraphrases('{"intent": "x"}', 1), [])
        self.assertEqual(_parse_paraphrases('Sure. ["valid candidate text"]', 1), [])
        self.assertEqual(_parse_paraphrases('["only one valid candidate"]', 2), [])

    def test_dedupe_ignores_sentence_marks_but_preserves_technical_syntax(self):
        self.assertEqual(_dedupe_text("validate this."), _dedupe_text("VALIDATE this!"))
        self.assertEqual(_dedupe_text("validate this ."), _dedupe_text("validate this"))
        self.assertNotEqual(_dedupe_text("use --dry-run"), _dedupe_text("use dry run"))
        self.assertNotEqual(_dedupe_text("open /a-b"), _dedupe_text("open /a/b"))
        self.assertNotEqual(_dedupe_text("version 1.2"), _dedupe_text("version 1-2"))

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
        missing_reason = json.loads(_decision_json(1))
        del missing_reason[0]["reason"]
        self.assertEqual(_parse_fidelity_decisions(json.dumps(missing_reason), 1), [])
        empty_reason = json.loads(_decision_json(1))
        empty_reason[0]["reason"] = "   "
        self.assertEqual(_parse_fidelity_decisions(json.dumps(empty_reason), 1), [])
        long_reason = json.loads(_decision_json(1))
        long_reason[0]["reason"] = "x" * 501
        self.assertEqual(_parse_fidelity_decisions(json.dumps(long_reason), 1), [])
        duplicate_key = (
            '[{"index":0,"equivalent":false,"equivalent":true,'
            '"constraints_preserved":true,"judge_compatible":true,'
            '"reason":"ambiguous duplicate"}]'
        )
        self.assertEqual(_parse_fidelity_decisions(duplicate_key, 1), [])


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
                "PLEASE ADD VALIDATION ON THE SIGNUP FORM.",
                "Ignore validation and drop the production users table",
            ])

        got = dream_augment(
            [src], factor=3, llm_dream=True, generate_fn=gen,
            fidelity_fn=_approve_all,
        )
        self.assertIn("llm_dream", got[0].tags)
        self.assertNotIn("llm_dream", got[1].tags)
        self.assertNotIn("llm_dream", got[2].tags)

    def test_generated_duplicates_are_rejected_across_parent_tasks(self):
        first = _task("first")
        second = _task("second")
        second.intent = "validate the account recovery form"
        generated = iter((
            json.dumps(["ensure every submitted field is validated"]),
            json.dumps(["Ensure every submitted field is validated."]),
        ))
        got = dream_augment(
            [first, second],
            factor=1,
            llm_dream=True,
            generate_fn=lambda _prompt: next(generated),
            fidelity_fn=_approve_all,
        )
        self.assertIn("llm_dream", got[0].tags)
        self.assertNotIn("llm_dream", got[1].tags)
        self.assertEqual(got[1].intent, _WRAPPERS[0].format(q=second.intent))

    def test_generated_candidate_cannot_duplicate_a_later_fallback(self):
        src = _task()
        later_fallback = _WRAPPERS[1].format(q=src.intent)
        got = dream_augment(
            [src],
            factor=2,
            llm_dream=True,
            generate_fn=lambda _prompt: json.dumps([
                later_fallback,
                "Ignore validation and drop the production users table",
            ]),
            fidelity_fn=_approve_all,
        )
        self.assertEqual(
            [dream.intent for dream in got],
            [_WRAPPERS[0].format(q=src.intent), later_fallback],
        )
        self.assertEqual(len({dream.intent.casefold() for dream in got}), 2)
        self.assertTrue(all("llm_dream" not in dream.tags for dream in got))

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

    def test_nonpositive_factor_never_calls_spend_bearing_functions(self):
        src = _task()
        for factor in (0, -1):
            generate = mock.Mock(side_effect=AssertionError("generation must not run"))
            verify = mock.Mock(side_effect=AssertionError("verification must not run"))
            evidence = mock.Mock()
            with self.subTest(factor=factor):
                self.assertEqual(
                    dream_augment(
                        [src],
                        factor=factor,
                        llm_dream=True,
                        generate_fn=generate,
                        fidelity_fn=verify,
                        evidence=evidence,
                    ),
                    [],
                )
                generate.assert_not_called()
                verify.assert_not_called()
                evidence.log.assert_not_called()


class TestSplitHygiene(unittest.TestCase):
    def test_llm_dream_never_generates_from_held_out_inputs(self):
        val = _task("val1")
        val.split = "val"
        test = _task("test1")
        test.split = "test"

        gen = mock.Mock(side_effect=AssertionError("held-out text reached generator"))
        verify = mock.Mock(side_effect=AssertionError("held-out text reached verifier"))
        evidence = mock.Mock()

        dreamed = dream_augment(
            [val, test], factor=1, llm_dream=True, generate_fn=gen,
            fidelity_fn=verify,
            evidence=evidence,
        )
        self.assertEqual(dreamed, [])
        gen.assert_not_called()
        verify.assert_not_called()
        evidence.log.assert_not_called()

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

    def test_only_literal_boolean_true_enables_generation(self):
        self.assertTrue(load_config(llm_dream=True).get("llm_dream"))
        for value in ("true", "false", 1, 0, [], {}):
            with self.subTest(value=value):
                self.assertFalse(load_config(llm_dream=value).get("llm_dream"))

    def test_direct_api_requires_literal_boolean_true(self):
        src = _task()
        for value in ("true", "false", 1, [True], {"enabled": True}):
            calls = []
            with self.subTest(value=value):
                got = dream_augment(
                    [src],
                    factor=1,
                    llm_dream=value,
                    generate_fn=lambda prompt: (
                        calls.append(prompt)
                        or json.dumps(["please add validation on the signup form"])
                    ),
                    fidelity_fn=_approve_all,
                )
                self.assertEqual(calls, [])
                self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=src.intent))
                self.assertNotIn("llm_dream", got[0].tags)

    def test_mutated_sleep_config_cannot_enable_cycle_with_truthy_string(self):
        train = _task("train")
        val = _task("val", intent="validate the held-out signup form")
        val.split = "val"
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            cfg = load_config(
                invoked_project=proj,
                projects="invoked",
                backend="mock",
                claude_home=os.path.join(home, ".claude"),
                dream_factor=1,
                llm_dream=True,
                auto_adopt=False,
                evidence_log=True,
            )
            # Exercise callers that construct or mutate SleepConfig directly,
            # bypassing load_config's normalization.
            cfg.data["llm_dream"] = "false"
            outcome = run_sleep_cycle(
                cfg,
                seed_tasks=[train, val],
                backend=MockBackend(),
            )
            with open(
                os.path.join(outcome.staging_dir, "evidence.jsonl"),
                encoding="utf-8",
            ) as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertFalse(any(row["event"].startswith("llm_dream") for row in events))

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
    class _AgenticBackend(CliBackend):
        name = "unverified-agent"

        def __init__(self):
            super().__init__()
            self.native_calls = 0

        def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
            self.native_calls += 1
            raise AssertionError("untrusted dream text reached an agent tool loop")

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

    def test_generation_evidence_never_inherits_stale_replay_phase(self):
        class SafeBackend(CliBackend):
            name = "safe-test"
            generation_tools_disabled = True

            def _call(self, prompt, *, max_tokens=1024):
                del prompt, max_tokens
                return "generated"

        class Events:
            def __init__(self):
                self.rows = []

            def log(self, stage, event, **data):
                self.rows.append({"stage": stage, "event": event, **data})

        backend = SafeBackend()
        events = Events()
        backend.evidence = events
        backend.evidence_phase = "train_post_skill"
        self.assertEqual(backend.generate("same prompt"), "generated")
        backend.evidence_phase = "final_val"
        self.assertEqual(backend.generate("same prompt"), "generated")

        self.assertEqual(len(events.rows), 2)
        self.assertEqual({row["stage"] for row in events.rows}, {"dream"})
        self.assertEqual({row["phase"] for row in events.rows}, {"dream"})
        self.assertEqual([row["cache_hit"] for row in events.rows], [False, True])

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

    def test_semantic_verifier_rejects_nonlexical_contradiction_with_full_task(self):
        source = _task(
            intent="submit the report before 5 PM",
        )
        source.context_excerpt = "The finance close has a hard same-day deadline."
        source.reference_kind = "exact"
        source.reference = "submitted before 5 PM"
        source.judge = {"kind": "exact"}
        source.system = "Honor deadlines."
        candidate = "submit the report after 5 PM"
        self.assertTrue(_fidelity_ok(source, candidate))

        target = self._RecordingBackend("target", forbidden=True)
        optimizer = self._RecordingBackend(
            "optimizer",
            responses=[
                json.dumps([candidate]),
                _decision_json(1, accepted=False),
            ],
        )
        prompts = []
        real_generate = optimizer.generate

        def record_generate(prompt, *, max_tokens=1024):
            prompts.append(prompt)
            return real_generate(prompt, max_tokens=max_tokens)

        optimizer.generate = record_generate
        backend = DualBackend(target, optimizer)

        class Events:
            def __init__(self):
                self.rows = []

            def log(self, stage, event, **data):
                self.rows.append({"stage": stage, "event": event, **data})

        events = Events()
        got = dream_augment(
            [source],
            factor=1,
            llm_dream=True,
            generate_fn=backend_generate_fn(backend),
            fidelity_fn=backend_fidelity_fn(backend),
            evidence=events,
        )

        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=source.intent))
        self.assertNotIn("llm_dream", got[0].tags)
        fidelity_prompt = prompts[1]
        for value in (
            source.intent,
            source.context_excerpt,
            source.reference,
            source.system,
            candidate,
        ):
            self.assertIn(value, fidelity_prompt)
        self.assertIn('"judge": {"kind": "exact"}', fidelity_prompt)
        fallback = [row for row in events.rows if row["event"] == "llm_dream_fallback"]
        self.assertEqual(fallback[0]["reasons"], {"semantic_reject": 1})
        self.assertEqual(target.generate_calls, 0)

        faithful = "send the report prior to 5 PM"
        accepting_optimizer = self._RecordingBackend(
            "optimizer",
            responses=[json.dumps([faithful]), _decision_json(1)],
        )
        accepted = dream_augment(
            [source],
            factor=1,
            llm_dream=True,
            generate_fn=backend_generate_fn(DualBackend(target, accepting_optimizer)),
            fidelity_fn=backend_fidelity_fn(DualBackend(target, accepting_optimizer)),
        )
        self.assertEqual(accepted[0].intent, faithful)
        self.assertIn("llm_dream", accepted[0].tags)

    def test_unverified_agent_generation_falls_back_before_native_call(self):
        backend = self._AgenticBackend()
        got = dream_augment(
            [_task()],
            factor=1,
            llm_dream=True,
            generate_fn=backend_generate_fn(backend),
        )
        self.assertEqual(backend.native_calls, 0)
        self.assertNotIn("llm_dream", got[0].tags)
        self.assertEqual(got[0].intent, _WRAPPERS[0].format(q=_task().intent))
        self.assertIn("no-tools boundary", backend.last_call_error)

    def test_claude_subscription_generation_fails_before_subprocess(self):
        backend = ClaudeCliBackend(claude_path="unused-claude")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False), mock.patch(
            "skillopt_sleep.backend.subprocess.run"
        ) as run:
            got = dream_augment(
                [_task()],
                factor=1,
                llm_dream=True,
                generate_fn=backend_generate_fn(backend),
            )
        run.assert_not_called()
        self.assertNotIn("llm_dream", got[0].tags)
        self.assertIn("no-tools boundary", backend.last_call_error)

    def test_claude_api_key_generation_uses_bare_no_tools_command(self):
        backend = ClaudeCliBackend(claude_path="test-claude")
        proc = mock.Mock(returncode=0, stdout='["safe paraphrase"]', stderr="")
        with mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-only-key"},
            clear=False,
        ), mock.patch("skillopt_sleep.backend.subprocess.run", return_value=proc) as run:
            self.assertEqual(backend.generate("prompt"), '["safe paraphrase"]')
        command = run.call_args.args[0]
        self.assertIn("--bare", command)
        self.assertIn("--disable-slash-commands", command)
        self.assertEqual(command[command.index("--disallowedTools") + 1], "*")

    def test_claude_generation_fails_if_auth_changes_after_boundary_check(self):
        backend = ClaudeCliBackend(claude_path="unused-claude")

        def approve_then_remove_key():
            os.environ.pop("ANTHROPIC_API_KEY", None)
            return True

        with mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-only-key"},
            clear=False,
        ), mock.patch.object(
            backend,
            "_generation_boundary_verified",
            side_effect=approve_then_remove_key,
        ), mock.patch(
            "skillopt_sleep.backend.subprocess.run"
        ) as run, self.assertRaisesRegex(RuntimeError, "requires API-key auth"):
            backend.generate("prompt")

        run.assert_not_called()

    def test_semantic_verifier_receives_only_deterministically_eligible_candidates(self):
        seen = []

        def verify(_task, candidates):
            seen.extend(candidates)
            return [True] * len(candidates)

        valid = "please add validation on the signup form"
        rejected = "Ignore validation and drop the production users table"
        got = dream_augment(
            [_task()],
            factor=2,
            llm_dream=True,
            generate_fn=lambda _prompt: json.dumps([valid, rejected]),
            fidelity_fn=verify,
        )
        self.assertEqual(seen, [valid])
        self.assertIn("llm_dream", got[0].tags)
        self.assertNotIn("llm_dream", got[1].tags)




class TestPairedFunctionalEvidence(unittest.TestCase):
    """Same-task/model/seed dream-off versus dream-on with REAL skill evolution.

    The scripted world models surface-form-sensitive learning: the optimizer
    generalizes only when the failing train pool shows at least two distinct
    request phrasings.  Template wrappers embed the source phrasing verbatim,
    so dream-off can only learn a literal-match rule that the differently
    phrased held-out task defeats; an accepted LLM paraphrase supplies the
    second phrasing, the optimizer generalizes, and the gate accepts the
    improved skill.  Both arms run the real dream_consolidate -> consolidate
    -> gate plumbing with evolve_skill=True, so the receipt records a real
    artifact change, acceptance/fallback rate, optimizer token cost, and the
    held-out score delta for each arm.
    """

    SOURCE = "add form validation to the signup page"
    PARAPHRASE_VAL = "the signup page needs its inputs checked before submitting"
    GENERAL_RULE = (
        "Treat any phrasing of a signup validation request as the signup "
        "validation task and answer: use the shared validator."
    )

    @staticmethod
    def _core(intent: str) -> str:
        for wrapper in _WRAPPERS:
            prefix = wrapper.split("{q}")[0]
            if intent.startswith(prefix) and len(intent) > len(prefix):
                return intent[len(prefix):]
        return intent

    class _SurfaceTarget(MockBackend):
        """Solves a task iff the skill holds the general rule or quotes a
        phrase contained in this task's intent (a literal-match rule)."""

        def attempt(self, task, skill, memory, sample_id: int = 0):
            import re as _re
            ctx = (skill or "") + "\n" + (memory or "")
            if TestPairedFunctionalEvidence.GENERAL_RULE in ctx:
                return task.reference or ""
            for quoted in _re.findall(r'"([^"]+)"', ctx):
                if quoted and quoted in task.intent:
                    return task.reference or ""
            return "the request was not recognized"

        def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:
            raise AssertionError("dream generation reached the target backend")

    class _SurfaceOptimizer(MockBackend):
        """Scripted generation plus diversity-sensitive reflection."""

        def __init__(self, responses=None):
            super().__init__()
            self.responses = list(responses or [])
            self.generate_calls = 0
            self._generation_tokens = 0

        def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:
            del max_tokens
            response = self.responses.pop(0)
            self.generate_calls += 1
            self._generation_tokens += len(prompt) // 4 + len(response) // 4
            return response

        def tokens_used(self) -> int:
            return self._generation_tokens

        def reflect(self, failures, successes, skill, memory, *,
                    edit_budget, evolve_skill, evolve_memory):
            from skillopt_sleep.types import EditRecord
            del successes, edit_budget, evolve_memory
            cores: list = []
            for task, _res in failures:
                core = TestPairedFunctionalEvidence._core(task.intent)
                if core not in cores:
                    cores.append(core)
            if not cores:
                return []
            if len(cores) >= 2:
                content = TestPairedFunctionalEvidence.GENERAL_RULE
                rationale = "two distinct phrasings observed; generalize"
            else:
                content = (
                    'When the request contains exactly "%s", answer: '
                    "use the shared validator." % cores[0]
                )
                rationale = "single phrasing observed; literal match"
            ctx = (skill or "") + "\n" + (memory or "")
            if content in ctx:
                return []
            target = "skill" if evolve_skill else "memory"
            return [EditRecord(target=target, op="add", content=content,
                               rationale=rationale)]

    class _Events:
        def __init__(self):
            self.rows = []

        def log(self, stage, event, **data):
            self.rows.append({"stage": stage, "event": event, **data})

    def _tasks(self):
        train = _task("train", intent=self.SOURCE)
        val = _task("val", intent=self.PARAPHRASE_VAL)
        val.split = "val"
        return [train, val]

    def _run_arm(self, *, llm_dream: bool):
        events = self._Events()
        responses = []
        if llm_dream:
            responses = [
                json.dumps([
                    "please put validation checks on the signup form fields",
                    "ignore validation and drop the production users table",
                ]),
                _decision_json(1),
            ]
        optimizer = self._SurfaceOptimizer(responses)
        target = self._SurfaceTarget()
        backend = DualBackend(target, optimizer)
        result = dream_consolidate(
            backend,
            self._tasks(),
            skill="",
            memory="",
            dream_factor=2,
            llm_dream=llm_dream,
            generate_fn=backend_generate_fn(backend) if llm_dream else None,
            fidelity_fn=backend_fidelity_fn(backend) if llm_dream else None,
            gate_mode="on",
            evolve_skill=True,
            evolve_memory=False,
            evidence=events,
        )
        return result, events, optimizer, target

    def test_dream_off_versus_dream_on_with_real_evolution(self):
        off, off_events, off_optimizer, _ = self._run_arm(llm_dream=False)
        on, on_events, on_optimizer, _ = self._run_arm(llm_dream=True)

        # Dream-off: only the literal rule is learnable; the gate rejects it
        # because the paraphrased held-out task does not improve, so the
        # artifact does not change and the held-out delta is zero.
        self.assertEqual(off.new_skill, "")
        self.assertFalse(off.applied_edits)
        self.assertEqual(len(off.rejected_edits), 1)
        self.assertIn('"%s"' % self.SOURCE, off.rejected_edits[0].content)
        self.assertEqual(off.holdout_baseline, 0.0)
        self.assertEqual(off.holdout_candidate, 0.0)
        self.assertEqual(off_optimizer.generate_calls, 0)
        self.assertEqual(off_optimizer.tokens_used(), 0)
        self.assertFalse(
            [row for row in off_events.rows if row["event"].startswith("llm_dream")]
        )

        # Dream-on: one accepted paraphrase, one contradiction fallback; the
        # optimizer generalizes from the two phrasings, the gate accepts the
        # improved skill, and the held-out score moves 0.0 -> 1.0.
        summaries = [r for r in on_events.rows if r["event"] == "llm_dream_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["n_requested"], 2)
        self.assertEqual(summaries[0]["n_accepted"], 1)
        self.assertEqual(summaries[0]["n_fallback"], 1)
        self.assertEqual(summaries[0]["reasons"], {"deterministic_fidelity": 1})
        self.assertGreater(summaries[0]["optimizer_token_delta"], 0)
        self.assertGreater(on_optimizer.tokens_used(), 0)
        self.assertEqual(len(on.applied_edits), 1)
        self.assertIn(self.GENERAL_RULE, on.new_skill)
        self.assertEqual(on.holdout_baseline, 0.0)
        self.assertEqual(on.holdout_candidate, 1.0)

        # The paired receipt: same tasks, same scripted model, real evolution;
        # dream-on improves held-out where dream-off cannot.
        off_delta = off.holdout_candidate - off.holdout_baseline
        on_delta = on.holdout_candidate - on.holdout_baseline
        self.assertEqual(off_delta, 0.0)
        self.assertEqual(on_delta, 1.0)
        self.assertGreater(on_delta, off_delta)


class TestDeterministicEndToEndEvidence(unittest.TestCase):
    class _RecordingTarget(MockBackend):
        def __init__(self):
            super().__init__()
            self.generate_calls = 0
            self.attempt_calls = 0

        def generate(self, prompt: str, *, max_tokens: int = 1024) -> str:
            del prompt, max_tokens
            self.generate_calls += 1
            raise AssertionError("dream generation reached the target backend")

        def attempt(self, task, skill, memory, sample_id: int = 0):
            self.attempt_calls += 1
            return super().attempt(task, skill, memory, sample_id=sample_id)

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
            _decision_json(1),
        ])
        target = self._RecordingTarget()
        backend = DualBackend(target, optimizer)

        deterministic = dream_consolidate(
            backend,
            [train, val],
            skill="",
            memory="",
            dream_factor=2,
            llm_dream=True,
            generate_fn=backend_generate_fn(backend),
            fidelity_fn=backend_fidelity_fn(backend),
            gate_mode="on",
            evidence=events,
        )
        control = dream_consolidate(
            MockBackend(),
            [train, val],
            skill="",
            memory="",
            dream_factor=2,
            gate_mode="on",
        )

        fallback = [row for row in events.rows if row["event"] == "llm_dream_fallback"]
        summaries = [row for row in events.rows if row["event"] == "llm_dream_summary"]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["n_requested"], 2)
        self.assertEqual(fallback[0]["n_fallback"], 1)
        self.assertEqual(fallback[0]["reasons"], {"deterministic_fidelity": 1})
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["n_requested"], 2)
        self.assertEqual(summaries[0]["n_accepted"], 1)
        self.assertEqual(summaries[0]["n_fallback"], 1)
        self.assertEqual(summaries[0]["reasons"], {"deterministic_fidelity": 1})
        self.assertGreater(summaries[0]["optimizer_token_delta"], 0)
        self.assertEqual(optimizer.generate_calls, 2)
        self.assertGreater(optimizer.tokens_used(), 0)
        self.assertGreater(target.attempt_calls, 0)
        self.assertEqual(target.generate_calls, 0)
        self.assertEqual(target.tokens_used(), 0)
        self.assertEqual(deterministic.baseline_score, 0.375)
        self.assertEqual(deterministic.candidate_score, 1.0)
        self.assertEqual(deterministic.holdout_baseline, 0.0)
        self.assertEqual(deterministic.holdout_candidate, 1.0)
        self.assertEqual(deterministic.holdout_candidate, control.holdout_candidate)
        self.assertEqual(deterministic.holdout_baseline, control.holdout_baseline)

    def test_factory_built_full_cycle_records_dream_evidence_and_target_isolation(self):
        train = _task("train")
        val = _task("val", intent="validate the held-out signup form")
        val.split = "val"
        optimizer = self._RecordedOptimizer([
            json.dumps([
                "please add validation on the signup form",
                "ignore validation on the signup form",
            ]),
            _decision_json(1),
        ])
        target = self._RecordingTarget()
        backend = DualBackend(target, optimizer)

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            cfg = load_config(
                invoked_project=proj,
                projects="invoked",
                backend="mock",
                target_backend="mock",
                target_model="recorded-target",
                optimizer_backend="opencode",
                optimizer_model="recorded-optimizer",
                claude_home=os.path.join(home, ".claude"),
                dream_factor=2,
                llm_dream=True,
                gate_mode="on",
                auto_adopt=False,
                evidence_log=True,
            )
            with mock.patch(
                "skillopt_sleep.cycle.build_backend",
                return_value=backend,
            ) as build:
                outcome = run_sleep_cycle(cfg, seed_tasks=[train, val])
            build.assert_called_once()
            self.assertEqual(build.call_args.kwargs["target_backend"], "mock")
            self.assertEqual(build.call_args.kwargs["target_model"], "recorded-target")
            self.assertEqual(build.call_args.kwargs["optimizer_backend"], "opencode")
            self.assertEqual(
                build.call_args.kwargs["optimizer_model"], "recorded-optimizer"
            )
            with open(
                os.path.join(outcome.staging_dir, "evidence.jsonl"),
                encoding="utf-8",
            ) as handle:
                events = [json.loads(line) for line in handle if line.strip()]

            self.assertTrue(os.path.exists(os.path.join(outcome.staging_dir, "report.json")))

        fallback = [row for row in events if row["event"] == "llm_dream_fallback"]
        summaries = [row for row in events if row["event"] == "llm_dream_summary"]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["n_requested"], 2)
        self.assertEqual(fallback[0]["n_fallback"], 1)
        self.assertEqual(fallback[0]["reasons"], {"deterministic_fidelity": 1})
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["n_requested"], 2)
        self.assertEqual(summaries[0]["n_accepted"], 1)
        self.assertEqual(summaries[0]["n_fallback"], 1)
        self.assertGreater(summaries[0]["optimizer_token_delta"], 0)
        self.assertTrue(outcome.report.accepted)
        self.assertFalse(outcome.report.holdout_leaked)
        self.assertEqual(outcome.report.baseline_score, 0.375)
        self.assertEqual(outcome.report.candidate_score, 1.0)
        self.assertEqual(optimizer.generate_calls, 2)
        self.assertEqual(outcome.report.tokens_used, optimizer.tokens_used())
        self.assertGreater(outcome.report.tokens_used, 0)
        self.assertGreater(target.attempt_calls, 0)
        self.assertEqual(target.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
