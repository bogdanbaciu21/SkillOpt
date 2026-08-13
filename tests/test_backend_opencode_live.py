"""Opt-in live tests for the OpenCode CLI backend.

These tests make real model calls through the user's OpenCode installation,
account, and configuration.  They run unchanged on Windows and POSIX, but are
skipped unless ``SKILLOPT_TEST_REAL_OPENCODE=1`` is set.  An opted-in run also
requires an explicit model in ``SKILLOPT_SLEEP_OPENCODE_MODEL`` so the
provider/model selection is explicit.  They may incur provider charges and
create entries in the user's OpenCode session history.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skillopt_sleep.backend import _NO_WINDOW, OpenCodeCliBackend, resolve_opencode_path
from skillopt_sleep.config import DEFAULTS, SleepConfig
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.evidence import read_events
from skillopt_sleep.types import TaskRecord

_LIVE_ENABLED = os.environ.get("SKILLOPT_TEST_REAL_OPENCODE", "").strip() == "1"
_PLAIN_MARKER = "SKILLOPT_OPENCODE_PLAIN_OK_7F3C"
_CYCLE_MARKER = "SKILLOPT_OPENCODE_CYCLE_OK_9A6D"

pytestmark = pytest.mark.skipif(
    not _LIVE_ENABLED,
    reason="set SKILLOPT_TEST_REAL_OPENCODE=1 to make real OpenCode model calls",
)


def _live_settings() -> tuple[str, str]:
    """Return the model and executable required by the opted-in tests."""
    model = os.environ.get("SKILLOPT_SLEEP_OPENCODE_MODEL", "").strip()
    if not model:
        pytest.fail(
            "SKILLOPT_SLEEP_OPENCODE_MODEL is required for opted-in OpenCode live tests",
            pytrace=False,
        )

    opencode_path = resolve_opencode_path()
    if shutil.which(opencode_path) is None:
        pytest.fail(
            "OpenCode CLI was not found for an opted-in live test",
            pytrace=False,
        )
    return model, opencode_path


def _require_marker(response: str, marker: str, backend: OpenCodeCliBackend) -> None:
    """Fail without copying provider output into pytest's assertion report."""
    if backend.last_call_error:
        pytest.fail("the real OpenCode call failed", pytrace=False)
    if not response:
        pytest.fail("the real OpenCode call returned no text", pytrace=False)
    if marker not in response:
        pytest.fail("the real OpenCode response did not contain the requested marker", pytrace=False)


def _add_mcp_canary(monkeypatch, tmp_path: Path) -> tuple[Path, str]:
    """Add a local MCP that leaves a marker if OpenCode starts it."""
    mcp_name = f"skillopt-live-canary-{secrets.token_hex(8)}"
    marker = tmp_path / "mcp-started.txt"
    script = tmp_path / "mcp-canary.py"
    script.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('started', encoding='utf-8')\n",
        encoding="utf-8",
    )
    config = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            mcp_name: {
                "type": "local",
                "command": [sys.executable, str(script)],
            }
        },
    }

    if not os.environ.get("OPENCODE_CONFIG"):
        config_path = tmp_path / "mcp-canary-opencode.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("OPENCODE_CONFIG", str(config_path))
        return marker, mcp_name

    if not os.environ.get("OPENCODE_CONFIG_DIR"):
        config_dir = tmp_path / "mcp-canary-config"
        config_dir.mkdir()
        (config_dir / "opencode.json").write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(config_dir))
        return marker, mcp_name

    pytest.fail(
        "the live MCP canary needs either OPENCODE_CONFIG or OPENCODE_CONFIG_DIR to be unused",
        pytrace=False,
    )


def _require_mcp_canary_configured(opencode_path: str, mcp_name: str, tmp_path: Path) -> None:
    """Check the synthetic MCP is visible without starting MCP services."""
    env = os.environ.copy()
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    env["OPENCODE_PURE"] = "1"
    env["PWD"] = str(tmp_path)
    env.pop("OLDPWD", None)
    try:
        proc = subprocess.run(
            [opencode_path, "debug", "config", "--pure"],
            capture_output=True,
            creationflags=_NO_WINDOW,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            cwd=tmp_path,
            env=env,
        )
        resolved = json.loads(proc.stdout or "") if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RecursionError):
        resolved = None
    mcp = resolved.get("mcp") if isinstance(resolved, dict) else None
    entry = mcp.get(mcp_name) if isinstance(mcp, dict) else None
    if not isinstance(entry, dict) or entry.get("enabled") is False:
        pytest.fail("the real OpenCode config did not include the MCP canary", pytrace=False)


def test_real_opencode_plain_replay(monkeypatch, tmp_path):
    """Exercise one public backend attempt against the real OpenCode CLI."""
    model, opencode_path = _live_settings()
    monkeypatch.setenv("SKILLOPT_SLEEP_PROMPTS_PATH", str(tmp_path / "no-prompt-overrides.json"))
    mcp_marker, mcp_name = _add_mcp_canary(monkeypatch, tmp_path)
    _require_mcp_canary_configured(opencode_path, mcp_name, tmp_path)

    backend = OpenCodeCliBackend(model=model, opencode_path=opencode_path)
    task = TaskRecord(
        id="opencode-live-plain",
        project=str(tmp_path),
        intent=f"Reply with exactly this text and nothing else: {_PLAIN_MARKER}",
        reference_kind="exact",
        reference=_PLAIN_MARKER,
    )

    response = backend.attempt(task, skill="", memory="")

    if mcp_marker.exists():
        pytest.fail("the real OpenCode call started a configured MCP server", pytrace=False)
    _require_marker(response, _PLAIN_MARKER, backend)


def test_real_opencode_cycle_smoke(monkeypatch, tmp_path):
    """Run one seeded sleep cycle through a real, factory-built OpenCode backend."""
    model, opencode_path = _live_settings()
    project = tmp_path / "project"
    state_dir = tmp_path / "state"
    claude_home = tmp_path / "claude-home"
    project.mkdir()
    monkeypatch.setenv("SKILLOPT_SLEEP_PROMPTS_PATH", str(tmp_path / "no-prompt-overrides.json"))
    monkeypatch.setenv("SKILLOPT_SLEEP_WORKERS", "1")

    cfg = SleepConfig(
        data={
            **DEFAULTS,
            "backend": "opencode",
            "model": model,
            "opencode_path": opencode_path,
            "projects": "invoked",
            "invoked_project": str(project),
            "state_dir": str(state_dir),
            "claude_home": str(claude_home),
            "gate_mode": "on",
            "evolve_skill": False,
            "evolve_memory": False,
            "llm_mine": False,
            "dream_rollouts": 1,
            "dream_factor": 0,
            "recall_k": 0,
            "multi_skill_report": False,
            "auto_adopt": False,
            "evidence_log": True,
            "redact_secrets": True,
            "progress": False,
        }
    )
    task = TaskRecord(
        id="opencode-live-cycle",
        project=str(project),
        intent=f"Reply with exactly this text and nothing else: {_CYCLE_MARKER}",
        reference_kind="exact",
        reference=_CYCLE_MARKER,
        split="val",
    )

    outcome = run_sleep_cycle(cfg, seed_tasks=[task])

    staging_dir = Path(outcome.staging_dir)
    try:
        staging_dir.resolve().relative_to(project.resolve())
    except (OSError, ValueError):
        pytest.fail("the live cycle did not stage inside its temporary project", pytrace=False)
    artifact_paths = {name: staging_dir / name for name in ("report.json", "diagnostics.json", "evidence.jsonl")}
    if not all(path.is_file() for path in artifact_paths.values()):
        pytest.fail("the live cycle did not write its expected staging artifacts", pytrace=False)

    try:
        with artifact_paths["diagnostics.json"].open(encoding="utf-8") as handle:
            diagnostics = json.load(handle)
    except (OSError, json.JSONDecodeError):
        pytest.fail("the live cycle wrote unreadable diagnostics", pytrace=False)
    if not isinstance(diagnostics, dict):
        pytest.fail("the live cycle wrote invalid diagnostics", pytrace=False)
    holdout_detail = diagnostics.get("holdout_detail")
    holdout_entry = (
        holdout_detail[0]
        if isinstance(holdout_detail, list) and len(holdout_detail) == 1 and isinstance(holdout_detail[0], dict)
        else {}
    )
    response_len = holdout_entry.get("response_len", 0)
    if (
        diagnostics.get("backend") != "opencode"
        or diagnostics.get("call_error")
        or not holdout_entry
        or holdout_entry.get("hard") != 1.0
        or not isinstance(response_len, (int, float))
        or response_len <= 0
    ):
        pytest.fail("the live cycle diagnostics did not record a successful replay", pytrace=False)

    events = read_events(str(artifact_paths["evidence.jsonl"]))
    uncached_model_calls = [
        event
        for event in events
        if event.get("stage") == "replay" and event.get("event") == "model_call" and not event.get("cache_hit")
    ]
    model_calls = [
        event
        for event in events
        if event.get("stage") == "replay"
        and event.get("event") == "model_call"
        and event.get("kind") == "attempt"
        and not event.get("cache_hit")
    ]
    replay_results = [
        event
        for event in events
        if event.get("stage") == "replay" and event.get("event") == "result" and event.get("task_id") == task.id
    ]
    cycle_ends = [event for event in events if event.get("stage") == "cycle" and event.get("event") == "end"]

    # Keep any provider text in the temporary evidence file and out of pytest output.
    if len(uncached_model_calls) != 1 or len(model_calls) != 1:
        pytest.fail(
            "the live cycle did not make exactly one uncached OpenCode model call",
            pytrace=False,
        )
    if _CYCLE_MARKER not in str(model_calls[0].get("response", "")):
        pytest.fail("the live cycle did not complete one successful OpenCode attempt", pytrace=False)
    replay_phases = {event.get("phase") for event in replay_results}
    if (
        len(replay_results) != 3
        or replay_phases != {"baseline_val", "train", "final_val"}
        or any(event.get("hard") != 1.0 for event in replay_results)
    ):
        pytest.fail("the live cycle did not score its seeded replay successfully", pytrace=False)
    if len(cycle_ends) != 1 or cycle_ends[0].get("outcome") != "completed":
        pytest.fail("the live cycle did not reach completion", pytrace=False)
    if outcome.report.n_tasks != 1 or outcome.report.n_replayed != 1:
        pytest.fail("the live cycle report did not record its seeded replay", pytrace=False)
    if not outcome.staging_dir or outcome.adopted or outcome.adopted_paths:
        pytest.fail("the live cycle did not preserve review-before-adopt behavior", pytrace=False)
