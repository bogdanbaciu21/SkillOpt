# Multi-skill staging and subset adoption

There are two layers here. Do not collapse them.

1. **Low-level adoption API** — `staged_skills()` / `adopt_skills()`, plus
   `skillopt-sleep status` and `skillopt-sleep adopt --skill`. This slice is
   complete: a night can stage one proposal file per resolved skill, a reviewer
   can list those names, and an explicit subset is copied over the live files
   with a backup and a hash receipt.
2. **Opt-in nightly fan-out** — with `multi_skill_report`, each hinted group
   resolves and reads *its own* live `SKILL.md`, consolidates from that baseline,
   and stages an independent proposal. Promotion remains a separate human
   decision; `auto_adopt` never applies the fan-out implicitly.

Nothing here changes a single-managed-skill night. If a night stages no per-skill
proposals, the staging directory and `manifest.json` are exactly the legacy ones
and `skillopt-sleep adopt` keeps working unchanged.

## Nightly wiring (`run_sleep_cycle`)

When `multi_skill_report` is on and hinted groups pass the gate:

- the managed catch-all is **not** staged as a per-skill proposal (it stays on
  `proposed_SKILL.md`);
- each hinted group name is resolved with `resolve_skill` against
  `skill_search_roots(cfg)` and its live document is read before consolidation;
- each proposal targets the same resolved path that supplied its baseline;
- only `FOUND` unique live paths become `SkillProposal` rows;
- missing, ambiguous, rejected, unreadable, empty, or colliding names are skipped
  rather than aborting the night, and each skip is recorded in both report
  formats.

Review remains explicit. `auto_adopt` still only runs the legacy `adopt()`
pair; it never silently promotes every staged skill.

## Staging layout

Legacy (single managed skill) — unchanged:

```text
.skillopt-sleep/staging/20260728-013000/
├── manifest.json          # live_skill_path, live_memory_path, has_skill, has_memory, accepted
├── proposed_SKILL.md
├── proposed_CLAUDE.md
├── report.json
└── report.md
```

Multi-skill night — one extra file and one manifest row per skill:

```text
.skillopt-sleep/staging/20260728-013000/
├── manifest.json          # …the legacy keys plus "skills": [ … ]
├── proposed_SKILL.alpha.md
├── proposed_SKILL.beta.md
├── report.json            # report.skill_groups carries each skill's gate evidence
└── report.md
```

```json
{
  "live_skill_path": "/home/dev/.claude/skills/alpha/SKILL.md",
  "has_skill": false,
  "accepted": true,
  "skills": [
    {
      "skill_name": "alpha",
      "proposed_file": "proposed_SKILL.alpha.md",
      "live_skill_path": "/home/dev/.claude/skills/alpha/SKILL.md",
      "sha256": "<sha256 of proposed_SKILL.alpha.md>"
    },
    {
      "skill_name": "beta",
      "proposed_file": "proposed_SKILL.beta.md",
      "live_skill_path": "/home/dev/.claude/skills/beta/SKILL.md",
      "sha256": "<sha256 of proposed_SKILL.beta.md>"
    }
  ]
}
```

A skill name must be a single safe path segment and a live path must be an
absolute, traversal-free `*.md` file; two skills may not share a name or a target
file. A refused fan-out writes no `manifest.json`, so the folder is not adoptable.

## Adopting a reviewed subset

Low-level API:

```python
from skillopt_sleep.staging import adopt_skills, latest_staging, staged_skills

staging = latest_staging("/path/to/project")
[row["skill_name"] for row in staged_skills(staging)]   # ['alpha', 'beta']

receipts = adopt_skills(staging, ["alpha"])             # beta is left alone
receipts[0].sha256_before, receipts[0].sha256_after
```

CLI:

```text
python -m skillopt_sleep status --project PATH
python -m skillopt_sleep adopt --project PATH --skill alpha
python -m skillopt_sleep adopt --project PATH --skill alpha --skill beta
python -m skillopt_sleep adopt --project PATH --all-skills
```

On a multi-skill night, bare `adopt` does **not** silently promote every staged
skill. It lists the names and asks for `--skill` or `--all-skills`. Legacy
nights (no `skills` in the manifest) still use `adopt()` unchanged.

- `skill_names=None` adopts every staged skill; `[]` adopts nothing.
- An unknown or repeated name, an empty `--skill` token, an unsafe manifest
  row, a missing proposal file, a sha256 mismatch, an empty proposal body, or a
  uniqueness / live-target collision raises `StagingError` **before** anything
  is written.
- Uniqueness and live-target checks run **at adoption time against every staged
  row**, not only the selection, so adopting one skill cannot hide a sibling
  that now points at the same file (including via casefold or realpath/symlink).
  A live path that exists as something other than a file is also refused.
- Each selected proposal is pinned by the manifest `sha256`. Tampering with the
  staged file, or dropping the pin, is refused with no writes.
- The live target must already be `<skill_name>/SKILL.md`. Adopt will not create
  parent directories, follow a symlink file, or write through a symlink parent.
- Each live file is backed up to `backup/skills/<skill>/` and written atomically.
- If any write fails — including `adopted_skills.json` — every live file in the
  selection is restored (and files that did not exist before are removed), and
  the previous receipt bytes are restored atomically, so a partial adoption
  never survives.
- Receipts (`skill_name`, `live_skill_path`, `sha256_before`, `sha256_after`,
  `backup_path`) are returned and written to `adopted_skills.json` in the staging
  directory. An empty `sha256_before` means the skill had no live file yet.

## Migrating

- **Consumers of `manifest.json`**: treat `"skills"` as optional; when absent the
  night is a legacy single-proposal one.
- **Consumers of `report.json`**: `skill_groups` is `[]` on a single-skill night,
  and the flat `accepted` / `gate_action` / score fields keep their meaning.
- **Adoption tooling**: `adopt()` still adopts the legacy single proposal pair.
  Use `adopt_skills()` for per-skill nights; the two are independent, and neither
  runs implicitly.
