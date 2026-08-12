"""Tests for explicit multi-skill subset adoption (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_adopt_skill_subset.py
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
    adopt_skills,
    staged_skills,
    write_staging,
)
from skillopt_sleep.types import SleepReport


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TwoSkillNight:
    """End-to-end fixture: a staged night with two per-skill proposals."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.live_root = os.path.join(tmp, "live")
        self.alpha_live = os.path.join(self.live_root, "alpha", "SKILL.md")
        self.beta_live = os.path.join(self.live_root, "beta", "SKILL.md")
        _write(self.alpha_live, "# alpha v1\n")
        _write(self.beta_live, "# beta v1\n")
        self.staging = write_staging(
            tmp,
            report=SleepReport(night=1, project=tmp, accepted=True),
            proposed_skill=None, proposed_memory=None,
            live_skill_path=self.alpha_live,
            live_memory_path=os.path.join(self.live_root, "CLAUDE.md"),
            report_md="# report\n",
            skill_proposals=[
                SkillProposal("alpha", "# alpha v2\n", self.alpha_live),
                SkillProposal("beta", "# beta v2\n", self.beta_live),
            ],
        )


class TestStagedSkills(unittest.TestCase):
    def test_rows_are_readable_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            rows = staged_skills(night.staging)
            self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])

    def test_legacy_single_proposal_night_has_no_staged_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_staging(
                tmp, report=SleepReport(night=1, project=tmp), proposed_skill="# s\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            self.assertEqual(staged_skills(out), [])
            self.assertEqual(adopt_skills(out), [])

    def test_malformed_skills_manifest_shape_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            for malformed in ({"not": "a list"}, [{"skill_name": "alpha"}, "bad"]):
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest["skills"] = malformed
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                with self.assertRaises(StagingError, msg=repr(malformed)):
                    staged_skills(night.staging)


class TestAdoptSkillSubset(unittest.TestCase):
    def test_adopting_one_skill_leaves_the_other_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipts = adopt_skills(night.staging, ["alpha"])
            self.assertEqual([r.skill_name for r in receipts], ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_receipts_carry_before_and_after_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipt = adopt_skills(night.staging, ["alpha"])[0]
            self.assertEqual(receipt.sha256_before, _sha("# alpha v1\n"))
            self.assertEqual(receipt.sha256_after, _sha("# alpha v2\n"))
            self.assertEqual(receipt.live_skill_path, night.alpha_live)
            self.assertEqual(_read(receipt.backup_path), "# alpha v1\n")

    def test_receipts_are_persisted_beside_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["beta"])
            with open(os.path.join(night.staging, "adopted_skills.json"),
                      encoding="utf-8") as f:
                rows = json.load(f)
            self.assertEqual([r["skill_name"] for r in rows], ["beta"])
            self.assertEqual(rows[0]["sha256_after"], _sha("# beta v2\n"))

    def test_selecting_no_skills_adopts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            self.assertEqual(adopt_skills(night.staging, []), [])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_selecting_every_skill_adopts_all_of_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            receipts = adopt_skills(night.staging)
            self.assertEqual([r.skill_name for r in receipts], ["alpha", "beta"])
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_a_new_live_file_reports_an_empty_before_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.beta_live)
            receipt = [r for r in adopt_skills(night.staging) if r.skill_name == "beta"][0]
            self.assertEqual(receipt.sha256_before, "")
            self.assertEqual(receipt.backup_path, "")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_unknown_or_repeated_selection_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            for selection in (["gamma"], ["alpha", "gamma"], ["alpha", "alpha"]):
                with self.assertRaises(StagingError, msg=str(selection)):
                    adopt_skills(night.staging, selection)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_missing_staged_proposal_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(os.path.join(night.staging, "proposed_SKILL.beta.md"))
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_unsafe_manifest_row_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = "relative/SKILL.md"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_manifest_proposal_filename_cannot_escape_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            outside = os.path.join(tmp, "outside.md")
            _write(outside, "# not a staged proposal\n")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][0]["proposed_file"] = os.path.relpath(
                outside, night.staging
            )
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")

    def test_adoption_preserves_existing_live_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.chmod(night.alpha_live, 0o640)
            adopt_skills(night.staging, ["alpha"])
            self.assertEqual(stat.S_IMODE(os.stat(night.alpha_live).st_mode), 0o640)

    def test_a_failed_write_rolls_the_whole_selection_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            # beta's live path becomes un-writable: its parent is now a file.
            os.unlink(night.beta_live)
            os.rmdir(os.path.dirname(night.beta_live))
            _write(os.path.dirname(night.beta_live), "not a directory\n")
            with self.assertRaises(OSError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_rollback_removes_files_that_did_not_exist_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.alpha_live)
            os.unlink(night.beta_live)
            os.rmdir(os.path.dirname(night.beta_live))
            _write(os.path.dirname(night.beta_live), "not a directory\n")
            with self.assertRaises(OSError):
                adopt_skills(night.staging)
            self.assertFalse(os.path.exists(night.alpha_live))

    def test_adoption_never_happens_without_an_explicit_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertTrue(os.path.exists(
                os.path.join(night.staging, "proposed_SKILL.alpha.md")))

    def test_tampered_duplicate_live_paths_are_refused_at_adopt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = night.alpha_live
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertFalse(
                os.path.exists(os.path.join(night.staging, "adopted_skills.json")))

    def test_live_target_that_is_not_a_file_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.unlink(night.beta_live)
            os.mkdir(night.beta_live)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertTrue(os.path.isdir(night.beta_live))

    def test_receipt_write_failure_rolls_back_live_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            os.makedirs(os.path.join(night.staging, "adopted_skills.json"))
            with self.assertRaises(OSError):
                adopt_skills(night.staging, ["alpha"])
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")


class TestCycleStagesResolvedSkillSubset(unittest.TestCase):
    """run_sleep_cycle stages resolved skills; adopt promotes only the subset."""

    def _hinted_tasks(self):
        from dataclasses import replace

        from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
        from skillopt_sleep.mine import assign_splits

        research = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        programming = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
        tagged = [replace(t, skill_hint="research-skill") for t in research]
        tagged += [replace(t, id=f"prog-{t.id}", skill_hint="programming-skill")
                   for t in programming]
        return tagged

    def test_cycle_stages_both_skills_and_subset_adopt_touches_only_one(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md")
            programming_live = os.path.join(
                claude_home, "skills", "programming-skill", "SKILL.md")
            _write(research_live, "# research-skill v1\n")
            _write(programming_live, "# programming-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=False,
                multi_skill_report=True,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            rows = staged_skills(outcome.staging_dir)
            names = [r["skill_name"] for r in rows]
            self.assertIn("research-skill", names)
            self.assertIn("programming-skill", names)
            self.assertTrue(os.path.isfile(os.path.join(
                outcome.staging_dir, "proposed_SKILL.research-skill.md")))
            self.assertTrue(os.path.isfile(os.path.join(
                outcome.staging_dir, "proposed_SKILL.programming-skill.md")))
            self.assertEqual(_read(research_live), "# research-skill v1\n")
            self.assertEqual(_read(programming_live), "# programming-skill v1\n")

            receipts = adopt_skills(outcome.staging_dir, ["research-skill"])
            self.assertEqual([r.skill_name for r in receipts], ["research-skill"])
            self.assertNotEqual(_read(research_live), "# research-skill v1\n")
            self.assertEqual(_read(programming_live), "# programming-skill v1\n")


class TestAdoptSkillCli(unittest.TestCase):
    def _cli(self, argv):
        import contextlib
        import io

        from skillopt_sleep.__main__ import main

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = main(argv)
        return rc, stdout.getvalue()

    def test_status_lists_staged_skill_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "status", "--project", tmp, "--claude-home", claude_home, "--json",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(out)
            self.assertEqual(payload["staged_skills"], ["alpha", "beta"])
            self.assertEqual(payload["latest_staging"], night.staging)

    def test_bare_adopt_on_a_multi_skill_night_lists_and_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
            ])
            self.assertEqual(rc, 2)
            self.assertIn("--skill", out)
            self.assertIn("alpha", out)
            self.assertIn("beta", out)
            self.assertEqual(_read(os.path.join(tmp, "live", "alpha", "SKILL.md")),
                             "# alpha v1\n")

    def test_adopt_skill_flag_promotes_only_the_named_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_all_skills_flag_promotes_every_staged_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--all-skills",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_repeated_skill_flags_adopt_the_named_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha", "--skill", "beta",
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(night.beta_live), "# beta v2\n")

    def test_skill_and_all_skills_together_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha", "--all-skills",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("not both", out)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_legacy_night_bare_adopt_still_copies_the_managed_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "SKILL.md")
            memory = os.path.join(tmp, "live", "CLAUDE.md")
            _write(live, "# live v1\n")
            _write(memory, "# mem v1\n")
            write_staging(
                tmp, report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# live v2\n", proposed_memory="# mem v2\n",
                live_skill_path=live, live_memory_path=memory,
                report_md="# report\n",
            )
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
            ])
            self.assertEqual(rc, 0, out)
            self.assertEqual(_read(live), "# live v2\n")
            self.assertEqual(_read(memory), "# mem v2\n")

    def test_skill_flag_on_a_legacy_night_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "SKILL.md")
            _write(live, "# live v1\n")
            write_staging(
                tmp, report=SleepReport(night=1, project=tmp, accepted=True),
                proposed_skill="# live v2\n", proposed_memory=None,
                live_skill_path=live,
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            claude_home = os.path.join(tmp, ".claude")
            os.makedirs(claude_home, exist_ok=True)
            rc, out = self._cli([
                "adopt", "--project", tmp, "--claude-home", claude_home,
                "--skill", "alpha",
            ])
            self.assertEqual(rc, 2)
            self.assertIn("no per-skill", out)
            self.assertEqual(_read(live), "# live v1\n")


class TestAdoptTimeRevalidationMega(unittest.TestCase):
    def test_casefold_live_path_collision_is_refused_at_adopt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            live_root = os.path.dirname(os.path.dirname(night.alpha_live))
            manifest["skills"][1]["live_skill_path"] = os.path.join(
                live_root, "ALPHA", "SKILL.md")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_symlink_realpath_collision_is_refused_at_adopt(self):
        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            alias = os.path.join(night.live_root, "alias", "SKILL.md")
            os.makedirs(os.path.dirname(alias), exist_ok=True)
            try:
                os.symlink(night.alpha_live, alias)
            except OSError:
                self.skipTest("symlinks unavailable")
            manifest_path = os.path.join(night.staging, "manifest.json")
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["skills"][1]["live_skill_path"] = alias
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(StagingError):
                adopt_skills(night.staging)
            self.assertEqual(_read(night.alpha_live), "# alpha v1\n")
            self.assertEqual(_read(night.beta_live), "# beta v1\n")

    def test_receipt_write_failure_restores_a_previous_receipt(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            night = TwoSkillNight(tmp)
            adopt_skills(night.staging, ["alpha"])
            receipt_path = os.path.join(night.staging, "adopted_skills.json")
            previous = _read(receipt_path)
            real_write = staging_mod._write_atomic

            def boom(path, text):
                if os.path.basename(path) == "adopted_skills.json":
                    raise OSError("disk full")
                return real_write(path, text)

            with mock.patch.object(staging_mod, "_write_atomic", side_effect=boom):
                with self.assertRaises(OSError):
                    adopt_skills(night.staging, ["beta"])
            self.assertEqual(_read(night.beta_live), "# beta v1\n")
            self.assertEqual(_read(night.alpha_live), "# alpha v2\n")
            self.assertEqual(_read(receipt_path), previous)


def _accepted_group(name, body):
    from skillopt_sleep.consolidate import ConsolidationResult
    from skillopt_sleep.multi_skill import CONSOLIDATED, GroupConsolidation

    result = ConsolidationResult(
        accepted=True, gate_action="accept_new_best",
        baseline_score=0.1, candidate_score=0.2,
        new_skill=body, new_memory="",
        applied_edits=[], rejected_edits=[],
        holdout_baseline=0.1, holdout_candidate=0.2,
    )
    return GroupConsolidation(
        skill_name=name, status=CONSOLIDATED, result=result, n_tasks=2,
    )


class TestSkillProposalsFromGroups(unittest.TestCase):
    def test_skips_managed_catch_all_and_unresolved_names(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import _skill_proposals_from_groups

        with tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            live = os.path.join(claude_home, "skills", "research-skill", "SKILL.md")
            _write(live, "# research v1\n")
            cfg = load_config(
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned",
            )
            proposals = _skill_proposals_from_groups(
                cfg,
                {
                    "skillopt-sleep-learned": _accepted_group(
                        "skillopt-sleep-learned", "# managed v2\n"),
                    "research-skill": _accepted_group(
                        "research-skill", "# research v2\n"),
                    "ghost-skill": _accepted_group(
                        "ghost-skill", "# ghost v2\n"),
                },
                "skillopt-sleep-learned",
            )
            names = [p.skill_name for p in proposals]
            self.assertEqual(names, ["research-skill"])
            self.assertEqual(proposals[0].live_skill_path, os.path.realpath(live))
            self.assertEqual(proposals[0].proposed_skill, "# research v2\n")

    def test_skips_a_second_skill_that_resolves_to_the_same_live_file(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import _skill_proposals_from_groups

        with tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            skills = os.path.join(claude_home, "skills")
            research = os.path.join(skills, "research-skill")
            alias = os.path.join(skills, "alias-skill")
            _write(os.path.join(research, "SKILL.md"), "# research v1\n")
            try:
                os.symlink(research, alias)
            except OSError:
                self.skipTest("symlinks unavailable")
            cfg = load_config(claude_home=claude_home)
            proposals = _skill_proposals_from_groups(
                cfg,
                {
                    "research-skill": _accepted_group(
                        "research-skill", "# research v2\n"),
                    "alias-skill": _accepted_group(
                        "alias-skill", "# alias v2\n"),
                },
                "skillopt-sleep-learned",
            )
            self.assertEqual([p.skill_name for p in proposals], ["research-skill"])


class TestCycleStagingGaps(unittest.TestCase):
    def _hinted_tasks(self):
        from dataclasses import replace

        from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
        from skillopt_sleep.mine import assign_splits

        research = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
        programming = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
        tagged = [replace(t, skill_hint="research-skill") for t in research]
        tagged += [replace(t, id=f"prog-{t.id}", skill_hint="programming-skill")
                   for t in programming]
        return tagged

    def test_missing_live_skill_is_skipped_not_aborted(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md")
            _write(research_live, "# research-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=False,
                multi_skill_report=True,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            names = [r["skill_name"] for r in staged_skills(outcome.staging_dir)]
            self.assertEqual(names, ["research-skill"])
            self.assertFalse(os.path.isfile(os.path.join(
                outcome.staging_dir, "proposed_SKILL.programming-skill.md")))

    def test_report_off_stages_no_per_skill_proposals(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            _write(os.path.join(claude_home, "skills", "research-skill", "SKILL.md"),
                   "# research-skill v1\n")
            _write(os.path.join(claude_home, "skills", "programming-skill", "SKILL.md"),
                   "# programming-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=False,
                multi_skill_report=False,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            self.assertEqual(staged_skills(outcome.staging_dir), [])

    def test_auto_adopt_does_not_promote_per_skill_live_files(self):
        from skillopt_sleep.config import load_config
        from skillopt_sleep.cycle import run_sleep_cycle

        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            research_live = os.path.join(
                claude_home, "skills", "research-skill", "SKILL.md")
            programming_live = os.path.join(
                claude_home, "skills", "programming-skill", "SKILL.md")
            _write(research_live, "# research-skill v1\n")
            _write(programming_live, "# programming-skill v1\n")
            cfg = load_config(
                invoked_project=proj, projects="invoked", backend="mock",
                claude_home=claude_home,
                managed_skill_name="skillopt-sleep-learned", auto_adopt=True,
                multi_skill_report=True,
            )
            outcome = run_sleep_cycle(cfg, seed_tasks=self._hinted_tasks())
            self.assertEqual(_read(research_live), "# research-skill v1\n")
            self.assertEqual(_read(programming_live), "# programming-skill v1\n")
            names = [r["skill_name"] for r in staged_skills(outcome.staging_dir)]
            self.assertIn("research-skill", names)
            self.assertIn("programming-skill", names)


if __name__ == "__main__":
    unittest.main()
