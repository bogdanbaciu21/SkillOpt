"""SkillOpt-Sleep — dream + associative recall for nightly consolidation.

Two opt-in mechanisms (both default OFF, so the cycle is unchanged unless the
user enables them) that the deployment experiments validated:

  * dream rollouts  — run each task K times and learn from the good-vs-bad
    contrast (set ``dream_rollouts > 1``). Stronger signal than one failure.
  * associative recall — each night, pull the K past tasks most similar to
    tonight's new ones into the dream (set ``recall_k > 0``). Replays relevant
    experience without re-running the whole history.

``dream_consolidate`` wires recall + synthetic augmentation + multi-rollout
consolidation and is called by BOTH the shipped plugin cycle and the benchmark
experiment harness, so the reported numbers exercise the exact code the plugin
runs. Pure-stdlib, zero research/private dependency.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Callable, List, Optional, Sequence

from skillopt_sleep.consolidate import ConsolidationResult, consolidate
from skillopt_sleep.types import TaskRecord

GenerateFn = Callable[[str], str]
FidelityFn = Callable[[TaskRecord, Sequence[str]], Sequence[bool]]

# ── synthetic augmentation ("dream up" variants of today's tasks) ─────────────

_WRAPPERS = [
    "(quick one) {q}",
    "Please handle this request: {q}",
    "For the daily report: {q}",
]


def _template_intent(task: TaskRecord, k: int) -> str:
    return _WRAPPERS[k % len(_WRAPPERS)].format(q=task.intent)


def _parse_paraphrases(raw: str, n: int) -> List[str]:
    """Accept exactly one JSON array containing exactly ``n`` safe strings."""
    try:
        parsed = json.loads((raw or "").strip())
    except (TypeError, ValueError, RecursionError):
        return []
    if not isinstance(parsed, list) or len(parsed) != n:
        return []
    out: List[str] = []
    for item in parsed:
        if not isinstance(item, str):
            return []
        text = item.strip()
        if not 8 <= len(text) <= 4000:
            return []
        out.append(text)
    return out


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


_NEGATION_MARKERS = re.compile(
    r"\b(?:cannot|can't|do\s+not|don't|drop|ignore|never|no|not|omit|remove|skip|without)\b",
    re.IGNORECASE,
)


def _protected_literals(value: str) -> set[str]:
    """Extract explicit literals whose removal would change task constraints."""
    literals = set()
    patterns = (
        r"`([^`\n]{1,160})`",
        r"\"([^\"\n]{1,160})\"",
        r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9_-]*",
        r"\b\d+(?:\.\d+)?(?:%|ms|s|m|h|kb|mb|gb)?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value or "", re.IGNORECASE):
            literal = match.group(1) if match.lastindex else match.group(0)
            normalized = _canonical_text(literal)
            if normalized:
                literals.add(normalized)
    return literals


def _fidelity_ok(task: TaskRecord, paraphrase: str) -> bool:
    """Apply deterministic, fail-closed checks before semantic verification.

    This is deliberately necessary but not sufficient: accepted candidates
    must also pass the task-aware optimizer verifier supplied to
    :func:`dream_augment`.
    """
    text = (paraphrase or "").strip()
    src = (task.intent or "").strip()
    if len(text) < 8 or not src:
        return False
    if _canonical_text(text) == _canonical_text(src):
        return False
    if "return only a json array" in _canonical_text(text):
        return False
    if len(text) > max(4000, len(src) * 4):
        return False
    # Adding or removing explicit negation is a common semantic inversion.
    if bool(_NEGATION_MARKERS.search(src)) != bool(_NEGATION_MARKERS.search(text)):
        return False
    candidate = _canonical_text(text)
    if any(literal not in candidate for literal in _protected_literals(src)):
        return False
    return True


def _parse_fidelity_decisions(raw: str, n: int) -> List[bool]:
    """Parse exact, indexed semantic-verification decisions."""
    try:
        parsed = json.loads((raw or "").strip())
    except (TypeError, ValueError, RecursionError):
        return []
    if not isinstance(parsed, list) or len(parsed) != n:
        return []
    decisions: List[bool] = []
    required = {"index", "equivalent", "constraints_preserved", "judge_compatible"}
    allowed = required | {"reason"}
    for expected, item in enumerate(parsed):
        if not isinstance(item, dict) or not required.issubset(item) or set(item) - allowed:
            return []
        if type(item["index"]) is not int or item["index"] != expected:
            return []
        flags = [item[name] for name in ("equivalent", "constraints_preserved", "judge_compatible")]
        if any(type(flag) is not bool for flag in flags):
            return []
        if "reason" in item and not isinstance(item["reason"], str):
            return []
        decisions.append(all(flags))
    return decisions


def _dream_record(task: TaskRecord, k: int, intent: str, extra_tags: Optional[List[str]] = None) -> TaskRecord:
    tags = list(task.tags) + ["dream"]
    if extra_tags:
        tags.extend(extra_tags)
    return TaskRecord(
        id=f"{task.id}_dream{k}", project=task.project,
        intent=intent, context_excerpt=task.context_excerpt,
        reference_kind=task.reference_kind, reference=task.reference,
        judge=dict(task.judge), system=task.system,
        tags=tags, split="train",
        origin="dream", derived_from=task.id,
        skill_hint=task.skill_hint,
    )


def dream_augment(
    real_tasks: List[TaskRecord],
    *,
    factor: int = 1,
    llm_dream: bool = False,
    generate_fn: Optional[GenerateFn] = None,
    fidelity_fn: Optional[FidelityFn] = None,
    evidence=None,
) -> List[TaskRecord]:
    """Create synthetic TRAIN variants of real tasks (origin='dream').

    Default path is a light, deterministic rephrasing. Dream tasks are
    training-only: they carry split='train' and never enter the val/test
    slices the gate scores on.

    Opt-in ``llm_dream=True`` asks ``generate_fn`` for paraphrase-only
    rewrites (parent reference/judge copied unchanged). Any parse or
    fidelity failure falls back to the same wrappers as the default path,
    so a night can degrade but not break. Template mode (the default) is
    byte-identical to the pre-llm_dream implementation.
    """
    out: List[TaskRecord] = []
    use_llm = bool(llm_dream) and generate_fn is not None
    if llm_dream and generate_fn is None and evidence is not None:
        evidence.log(
            "dream", "llm_dream_fallback",
            reason="no_generate_fn", n_requested=max(0, factor),
        )
    for t in real_tasks:
        requested = max(0, factor)
        parsed: List[str] = []
        reasons: dict[str, int] = {}
        if use_llm:
            try:
                from skillopt_sleep import prompts as prompt_registry
                prompt = prompt_registry.render("llm_dream", {
                    "__INTENT__": t.intent,
                    "__N__": str(requested),
                    "__CONTEXT__": (t.context_excerpt or "")[:400],
                })
                parsed = _parse_paraphrases(generate_fn(prompt), requested)
            except Exception:
                parsed = []
                reasons["generation_error"] = requested
        if use_llm and not parsed and "generation_error" not in reasons:
            reasons["malformed_generation"] = requested

        deterministic_ok = [False] * requested
        seen = {_canonical_text(t.intent)}
        for k, candidate in enumerate(parsed):
            key = _canonical_text(candidate)
            if key in seen:
                reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                continue
            seen.add(key)
            if not _fidelity_ok(t, candidate):
                reasons["deterministic_fidelity"] = reasons.get("deterministic_fidelity", 0) + 1
                continue
            deterministic_ok[k] = True

        semantic_ok = [False] * requested
        verifier_complete = False
        if any(deterministic_ok):
            if fidelity_fn is None:
                reasons["missing_semantic_verifier"] = sum(deterministic_ok)
            else:
                try:
                    decisions = list(fidelity_fn(t, parsed))
                    if len(decisions) != requested or any(type(value) is not bool for value in decisions):
                        raise ValueError("invalid semantic verifier response")
                    semantic_ok = decisions
                    verifier_complete = True
                except Exception:
                    reasons["semantic_verifier_error"] = sum(deterministic_ok)
        n_ok = 0
        for k in range(requested):
            extra: Optional[List[str]] = None
            if (
                use_llm
                and k < len(parsed)
                and deterministic_ok[k]
                and semantic_ok[k]
            ):
                intent = parsed[k]
                extra = ["llm_dream"]
                n_ok += 1
            else:
                intent = _template_intent(t, k)
                if verifier_complete and k < len(parsed) and deterministic_ok[k] and not semantic_ok[k]:
                    reasons["semantic_reject"] = reasons.get("semantic_reject", 0) + 1
            out.append(_dream_record(t, k, intent, extra))
        if use_llm and n_ok < requested and evidence is not None:
            evidence.log(
                "dream", "llm_dream_fallback",
                task_id=t.id,
                n_fallback=requested - n_ok,
                n_requested=requested,
                reasons=reasons,
            )
    return out


def backend_generate_fn(backend) -> GenerateFn:
    """Return optimizer-side generation without entering target task replay."""
    def generate(prompt: str) -> str:
        return backend.generate(prompt, max_tokens=1024)
    return generate


def backend_fidelity_fn(backend) -> FidelityFn:
    """Build a full-task semantic verifier routed through the optimizer API."""
    def verify(task: TaskRecord, candidates: Sequence[str]) -> Sequence[bool]:
        from skillopt_sleep import prompts as prompt_registry

        task_payload = {
            "intent": task.intent,
            "context_excerpt": task.context_excerpt,
            "reference_kind": task.reference_kind,
            "reference": task.reference,
            "judge": task.judge,
            "system": task.system,
            "tags": task.tags,
        }
        prompt = prompt_registry.render("llm_dream_fidelity", {
            "__TASK_JSON__": json.dumps(task_payload, ensure_ascii=False, sort_keys=True),
            "__CANDIDATES_JSON__": json.dumps(list(candidates), ensure_ascii=False),
        })
        return _parse_fidelity_decisions(
            backend.generate(prompt, max_tokens=1024),
            len(candidates),
        )
    return verify


# ── associative recall (experience replay of similar past tasks) ──────────────

def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _normalize_split(value: str) -> str:
    return {"replay": "train", "holdout": "val"}.get(value, value)


def recall_similar(
    new_tasks: List[TaskRecord],
    history: List[TaskRecord],
    k: int,
    *,
    exclude_ids: Optional[set[str]] = None,
) -> List[TaskRecord]:
    """Return the ``k`` historical tasks most lexically similar to any of
    tonight's ``new_tasks`` (max Jaccard token overlap). Recalled tasks are
    returned as training material (split='train'); deterministic, stdlib-only.

    Archived val/test tasks are never recalled, and ``exclude_ids`` blocks
    tonight's held-out ids (and their ``derived_from`` sources) from re-entering
    the training pool.
    """
    if not history or k <= 0 or not new_tasks:
        return []
    blocked = set(exclude_ids or ())
    for t in new_tasks:
        blocked.add(t.id)
        if t.derived_from:
            blocked.add(t.derived_from)
    new_tok = [_tokens(t.intent) for t in new_tasks]
    scored = []
    for h in history:
        if h.id in blocked:
            continue
        if _normalize_split(h.split) in ("val", "test"):
            continue
        ht = _tokens(h.intent)
        if not ht:
            continue
        sim = max(((len(ht & nt) / len(ht | nt)) if (ht | nt) else 0.0) for nt in new_tok)
        scored.append((sim, h.id, h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for sim, _id, h in scored[:max(0, k)]:
        if sim <= 0.0:
            break
        # recall as training material; copy so the source archive is untouched
        out.append(TaskRecord(
            id=f"recall:{h.id}", project=h.project, intent=h.intent,
            context_excerpt=h.context_excerpt, reference_kind=h.reference_kind,
            reference=h.reference, judge=dict(h.judge), system=h.system,
            tags=list(h.tags) + ["recall"], split="train", origin="real",
            derived_from=h.id,
            skill_hint=h.skill_hint,
        ))
    return out


# ── the shared nightly consolidation step ─────────────────────────────────────

def dream_consolidate(
    backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    *,
    history_tasks: Optional[List[TaskRecord]] = None,
    recall_k: int = 0,
    dream_rollouts: int = 1,
    dream_factor: int = 0,
    edit_budget: int = 4,
    gate_metric: str = "mixed",
    gate_mixed_weight: float = 0.5,
    gate_no_regression: bool = False,
    gate_mode: str = "on",
    evolve_skill: bool = True,
    evolve_memory: bool = True,
    night: int = 1,
    llm_dream: bool = False,
    generate_fn: Optional[GenerateFn] = None,
    fidelity_fn: Optional[FidelityFn] = None,
    evidence=None,
) -> ConsolidationResult:
    """Recall similar past experience + dream synthetic variants, then run one
    gated consolidation epoch over the enlarged training pool.

    ``tasks`` is the split-tagged pool for tonight (train + val); recall and
    augmentation only enlarge the TRAIN split, so the val slice the gate scores
    on is never polluted. With ``recall_k=0`` and ``dream_rollouts=1`` (the
    defaults) this is exactly the previous single-shot ``consolidate``.
    """
    train = [t for t in tasks if t.split == "train"]
    enlarged = list(tasks)
    if recall_k > 0 and history_tasks:
        held_out_ids = {
            t.id for t in tasks if _normalize_split(t.split) in ("val", "test")
        }
        for t in tasks:
            if t.derived_from:
                held_out_ids.add(t.derived_from)
        enlarged += recall_similar(
            train, history_tasks, recall_k, exclude_ids=held_out_ids,
        )
    if dream_factor > 0:
        seed = [t for t in enlarged if t.split == "train" and t.origin != "dream"]
        enlarged += dream_augment(
            seed,
            factor=dream_factor,
            llm_dream=llm_dream,
            generate_fn=generate_fn,
            fidelity_fn=fidelity_fn,
            evidence=evidence,
        )
    return consolidate(
        backend, enlarged, skill, memory,
        edit_budget=edit_budget, gate_metric=gate_metric,
        gate_mixed_weight=gate_mixed_weight,
        gate_no_regression=gate_no_regression, gate_mode=gate_mode,
        rollouts_k=dream_rollouts, evolve_skill=evolve_skill,
        evolve_memory=evolve_memory, night=night,
    )
