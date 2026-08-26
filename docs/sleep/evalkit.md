# Paired A/B evalkit

Sleep contributors have a shared instrument for "condition B beats condition A":

```text
python -m skillopt_sleep.evalkit --manifest tasks.json --a cond_a.json --b cond_b.json
```

The kit pairs outcomes by task id, runs McNemar's test on binary successes, and
reports a percentile-bootstrap confidence interval on the success-rate delta.
It does not change the nightly gate.

## Inputs

- `--manifest`: JSON list of task ids, or `{"ids": [...]}` / `{"tasks": [{"id": ...}]}`.
- `--a` / `--b`: JSON objects mapping those same ids to `0`/`1` (or a list of
  per-seed `0`/`1` values). A wrapper `{"outcomes": {...}}` is also accepted.
- `--aa`: A/A identity smoke check (reuses `--a` as both conditions). Must not reject.
- `--allow-graded`: permit non-binary scores. McNemar is omitted; bootstrap only.
- `--boot`, `--seed`, `--alpha`, `--json`.

The id sets of the manifest, A, and B must be identical. Cross-manifest
comparisons are refused. Seed lists must be non-empty, scores must be finite
and in `[0, 1]`, `alpha` must be strictly between 0 and 1, and `--boot` must be
between 1 and 1,000,000. JSON output is strict and never emits NaN/Infinity.

## Multi-seed

When each task maps to a same-length list of seed repeats, the kit:

1. averages per task across seeds for the headline delta and bootstrap CI
2. resamples whole tasks, preserving the task as the independent cluster
3. omits McNemar rather than treating repeated seeds as independent samples
4. publishes the per-seed deltas plus their mean and sample sd

That is the house answer to single-seed noise (see issue #108 and the
single-seed warning in `RESULTS.md`).

## RESULTS cell replay

`tests/fixtures/evalkit/results_searchqa_nano_gated.json` replays the published
SearchQA / GPT-5.4-nano / gated / cumulative nights=5 cell (baseline 0.560,
after 0.679, Δ +11.9 on n=1400). Per-task pairs were not published, so the
replay uses a documented maximum-concordance reconstruction: the first
`round(n * rate)` tasks succeed in each condition. The harness recovers the
published delta; it does not claim to recover the original microdata.

## A/A checks

```text
python -m skillopt_sleep.evalkit --manifest tests/fixtures/evalkit/aa_manifest.json \
  --a tests/fixtures/evalkit/aa_outcomes.json --aa
```

The command above is an identity smoke check: identical conditions must report
delta 0, McNemar p_exact = 1, and a CI that includes 0. The test suite separately
runs a seeded null simulation with genuine discordant pairs and bounds the
empirical type-I-error rate; comparing one array with itself is not presented as
a statistical calibration.
