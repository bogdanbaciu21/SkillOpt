"""Tests for the OpenCode CLI backend."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from unittest import mock

import pytest

from skillopt_sleep import cycle
from skillopt_sleep.__main__ import _add_common, _cfg_from_args
from skillopt_sleep.backend import (
    _NO_WINDOW,
    DualBackend,
    MockBackend,
    OpenCodeCliBackend,
    _parse_opencode_jsonl_text,
    build_backend,
    get_backend,
    resolve_opencode_path,
)
from skillopt_sleep.config import DEFAULTS, SleepConfig, load_config


class _FakeProc:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _event(event_type: str, *, session: str = "session-1", part=None) -> str:
    event = {"type": event_type, "sessionID": session}
    if part is not None:
        event["part"] = part
    return json.dumps(event)


def _success_stream(*texts: str) -> str:
    lines = [
        _event("step_start", part={"type": "step-start"}),
        *(_event("text", part={"type": "text", "text": text}) for text in texts),
        _event("step_finish", part={"type": "step-finish"}),
    ]
    return "\n".join(lines)


def _resolved_mcp(*names: str, disabled: bool = False) -> str:
    mcp = {
        name: {
            "type": "local",
            "command": ["mcp-server"],
            **({"enabled": False} if disabled else {}),
        }
        for name in names
    }
    return json.dumps({"mcp": mcp})


def _successful_plain_results(*mcp_names: str, answer: str = "answer") -> list[_FakeProc]:
    return [
        _FakeProc(_resolved_mcp(*mcp_names)),
        _FakeProc(_resolved_mcp(*mcp_names, disabled=True)),
        _FakeProc(_success_stream(answer)),
    ]


def test_resolve_opencode_path_precedence(monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_OPENCODE_PATH", "env-opencode")
    with mock.patch("shutil.which", side_effect=lambda value: os.path.abspath(f"resolved-{value}")):
        assert resolve_opencode_path("explicit-opencode") == os.path.abspath("resolved-explicit-opencode")
        assert resolve_opencode_path() == os.path.abspath("resolved-env-opencode")


def test_resolve_opencode_path_falls_back_to_path_or_command(monkeypatch, tmp_path):
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_PATH", raising=False)
    executable = str(tmp_path / "opencode")
    with mock.patch("shutil.which", return_value=executable):
        assert resolve_opencode_path() == executable
    with mock.patch("shutil.which", return_value=None):
        assert resolve_opencode_path() == "opencode"


def test_resolve_opencode_path_anchors_relative_path_search_result(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_PATH", raising=False)
    resolved = os.path.join("tools", "opencode")

    with mock.patch("shutil.which", return_value=resolved):
        assert resolve_opencode_path("opencode") == os.path.abspath(resolved)


@pytest.mark.skipif(os.name != "nt", reason="Windows PATHEXT shim behavior")
def test_resolve_opencode_path_preserves_windows_cmd_shim(monkeypatch):
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_PATH", raising=False)
    executable = r"C:\npm\opencode.CMD"

    with mock.patch("shutil.which", return_value=executable):
        assert resolve_opencode_path() == executable


@pytest.mark.parametrize("which_result", [None, os.path.join("bin", "opencode")])
@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_resolve_opencode_path_anchors_relative_paths(monkeypatch, tmp_path, which_result, source):
    monkeypatch.chdir(tmp_path)
    relative = os.path.join("bin", "opencode")
    explicit = relative if source == "explicit" else ""
    if source == "environment":
        monkeypatch.setenv("SKILLOPT_SLEEP_OPENCODE_PATH", relative)

    with mock.patch("shutil.which", return_value=which_result):
        assert resolve_opencode_path(explicit) == os.path.abspath(relative)


def test_constructor_uses_explicit_or_environment_model(monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_OPENCODE_MODEL", "env/model")
    with mock.patch("shutil.which", return_value=None):
        assert OpenCodeCliBackend(model="explicit/model").model == "explicit/model"
        assert OpenCodeCliBackend().model == "env/model"


def test_parse_opencode_jsonl_collects_text_and_ignores_extensions():
    raw = "\n".join(
        [
            _event("step_start", part={"type": "step-start"}),
            _event("future_event"),
            _event("reasoning", part={"type": "reasoning", "text": "hidden"}),
            _event("text", part={"type": "text", "text": "first"}),
            "",
            _event("text", part={"type": "text", "text": "second"}),
            _event("step_finish", part={"type": "step-finish"}),
        ]
    )
    assert _parse_opencode_jsonl_text(raw) == ("first\nsecond", "")


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        ("not json", "malformed_jsonl"),
        ('{"value":' + "9" * 5000 + "}", "malformed_jsonl"),
        (json.dumps(["not", "an", "object"]), "invalid_event"),
        (json.dumps({"type": "text"}), "invalid_event"),
        (
            "\n".join(
                [
                    _event("step_start", session="a", part={"type": "step-start"}),
                    _event("step_finish", session="b", part={"type": "step-finish"}),
                ]
            ),
            "mixed_session",
        ),
        (_event("error"), "error_event"),
        (
            "\n".join(
                [
                    _event("step_start", part={"type": "step-start"}),
                    _event("tool_use", part={"type": "tool", "tool": "shell"}),
                ]
            ),
            "unexpected_tool_event",
        ),
        (_event("step_start", part={"type": "step-start"}), "incomplete_stream"),
        (_success_stream("  "), "empty_response"),
    ],
)
def test_parse_opencode_jsonl_rejects_invalid_streams(raw, expected_code):
    assert _parse_opencode_jsonl_text(raw) == ("", expected_code)


def test_call_uses_stdin_temp_workspace_and_user_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-provider-key")
    monkeypatch.setenv("HOME", "/home/example")
    monkeypatch.setenv("PWD", "/original/project")
    monkeypatch.setenv("OLDPWD", "/previous/project")
    inline_config = json.dumps(
        {
            "default_agent": "unsafe-user-agent",
            "agent": {"unsafe-user-agent": {"permission": {"bash": "allow", "edit": "allow"}}},
        }
    )
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", inline_config)
    config_path = str(tmp_path / "custom-opencode.json")
    config_dir = str(tmp_path / ".opencode")
    monkeypatch.setenv("OPENCODE_CONFIG", config_path)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", config_dir)
    for key in (
        "NO_COLOR",
        "OPENCODE_DISABLE_AUTOUPDATE",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS",
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        "OPENCODE_DISABLE_SHARE",
        "OPENCODE_DISABLE_TERMINAL_TITLE",
        "OPENCODE_PURE",
    ):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("OPENCODE_PERMISSION", '{"*":"allow"}')
    monkeypatch.delenv("OPENCODE_DISABLE_DEFAULT_PLUGINS", raising=False)
    captured = []
    results = iter(_successful_plain_results("alpha", "beta"))

    def fake_run(cmd, **kwargs):
        snapshot = dict(kwargs)
        snapshot["env"] = kwargs["env"].copy()
        captured.append((cmd, snapshot))
        assert os.path.isdir(kwargs["cwd"])
        assert os.listdir(kwargs["cwd"]) == []
        return next(results)

    executable = str(tmp_path / "opencode")
    be = OpenCodeCliBackend(
        model="provider/model",
        opencode_path=executable,
        timeout=37,
    )
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        assert be._call("do the thing") == "answer"

    assert len(captured) == 3
    discovery_cmd, discovery_call = captured[0]
    verification_cmd, verification_call = captured[1]
    cmd, run_call = captured[2]
    expected_child_env = {
        "NO_COLOR": "1",
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_TERMINAL_TITLE": "1",
        "OPENCODE_PERMISSION": '{"*":"deny"}',
        "OPENCODE_PURE": "1",
    }
    for _, call in captured:
        assert call["capture_output"] is True
        assert call["creationflags"] == _NO_WINDOW
        assert call["text"] is True
        assert call["encoding"] == "utf-8"
        assert call["errors"] == "replace"
        assert call["timeout"] == be.timeout
        assert call["cwd"] == run_call["cwd"]
        assert call["env"]["PWD"] == call["cwd"]
        assert "OLDPWD" not in call["env"]
        for key, value in expected_child_env.items():
            assert call["env"][key] == value
    assert discovery_cmd == [executable, "debug", "config", "--pure"]
    assert verification_cmd == discovery_cmd
    assert cmd[:5] == [
        executable,
        "run",
        "--pure",
        "--format",
        "json",
    ]
    agent_name = cmd[cmd.index("--agent") + 1]
    assert agent_name.startswith("skillopt-sleep-")
    assert agent_name != "unsafe-user-agent"
    assert cmd[cmd.index("--title") + 1] == "skillopt-sleep"
    assert cmd[cmd.index("--dir") + 1] == run_call["cwd"]
    assert cmd[cmd.index("--model") + 1] == "provider/model"
    assert run_call["input"] == "do the thing"
    assert "input" not in discovery_call
    assert "input" not in verification_call
    assert "do the thing" not in cmd
    assert run_call["env"]["OPENAI_API_KEY"] == "ambient-provider-key"
    assert run_call["env"]["HOME"] == "/home/example"
    assert run_call["env"]["PWD"] != "/original/project"
    assert run_call["env"]["OPENCODE_CONFIG"] == config_path
    assert run_call["env"]["OPENCODE_CONFIG_DIR"] == config_dir
    discovered = json.loads(discovery_call["env"]["OPENCODE_CONFIG_CONTENT"])
    assert "mcp" not in discovered
    injected = json.loads(run_call["env"]["OPENCODE_CONFIG_CONTENT"])
    assert injected == {
        "agent": {
            agent_name: {
                "mode": "primary",
                "permission": {"*": "deny"},
            }
        },
        "mcp": {
            "alpha": {"enabled": False},
            "beta": {"enabled": False},
        },
    }
    assert verification_call["env"]["OPENCODE_CONFIG_CONTENT"] == run_call["env"]["OPENCODE_CONFIG_CONTENT"]
    assert run_call["env"]["OPENCODE_CONFIG_CONTENT"] != inline_config
    assert "unsafe-user-agent" not in run_call["env"]["OPENCODE_CONFIG_CONTENT"]
    assert "OPENCODE_DISABLE_DEFAULT_PLUGINS" not in run_call["env"]
    assert not os.path.exists(run_call["cwd"])
    assert be.last_call_error == ""


def test_relative_opencode_config_paths_become_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_path = os.path.join("profiles", "opencode.json")
    config_dir = os.path.join("profiles", "opencode")
    monkeypatch.setenv("OPENCODE_CONFIG", config_path)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", config_dir)
    captured = []
    results = iter(_successful_plain_results())

    def fake_run(cmd, **kwargs):
        captured.append(kwargs["env"].copy())
        return next(results)

    be = OpenCodeCliBackend(opencode_path=str(tmp_path / "opencode"))
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        assert be._call("hello") == "answer"

    assert len(captured) == 3
    assert all(env["OPENCODE_CONFIG"] == os.path.abspath(config_path) for env in captured)
    assert all(env["OPENCODE_CONFIG_DIR"] == os.path.abspath(config_dir) for env in captured)


def test_empty_model_leaves_model_selection_to_opencode(monkeypatch):
    monkeypatch.delenv("SKILLOPT_SLEEP_OPENCODE_MODEL", raising=False)
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=_successful_plain_results(),
    ) as run:
        assert be._call("hello") == "answer"
    assert "--model" not in run.call_args_list[-1].args[0]


def test_each_call_uses_a_new_agent_name():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with (
        mock.patch("secrets.token_hex", side_effect=["a" * 32, "b" * 32]),
        mock.patch(
            "skillopt_sleep.backend.subprocess.run",
            side_effect=_successful_plain_results() + _successful_plain_results(),
        ) as run,
    ):
        assert be._call("first") == "answer"
        assert be._call("second") == "answer"

    commands = [call.args[0] for call in run.call_args_list if call.args[0][1] == "run"]
    assert commands[0][commands[0].index("--agent") + 1] == ("skillopt-sleep-" + "a" * 32)
    assert commands[1][commands[1].index("--agent") + 1] == ("skillopt-sleep-" + "b" * 32)


@pytest.mark.parametrize(
    ("side_effect", "return_value", "error_fragment"),
    [
        (subprocess.TimeoutExpired("opencode", 1), None, "timed out"),
        (OSError("secret path"), None, "could not be executed"),
        (None, _FakeProc("misleading", returncode=2), "exited 2"),
        (None, _FakeProc("not json"), "malformed JSONL"),
    ],
)
def test_call_records_process_and_protocol_failures(side_effect, return_value, error_fragment):
    be = OpenCodeCliBackend(opencode_path="opencode", timeout=1)
    run_result = side_effect if side_effect is not None else return_value
    effects = _successful_plain_results()[:2] + [run_result]
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=effects):
        assert be._call("hello") == ""
    assert error_fragment in be.last_call_error
    assert "secret path" not in be.last_call_error


def test_mcp_reenabled_by_final_config_stops_before_model_call():
    discovered = _FakeProc(_resolved_mcp("global-server"))
    reenabled = _FakeProc(
        json.dumps(
            {
                "mcp": {
                    "global-server": {
                        "type": "local",
                        "command": ["mcp-server"],
                        "enabled": True,
                    }
                }
            }
        )
    )
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[discovered, reenabled],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert all(call.args[0][1:3] == ["debug", "config"] for call in run.call_args_list)
    assert "disable every configured MCP server" in be.last_call_error


def test_new_enabled_mcp_in_final_config_stops_before_model_call():
    discovered = _FakeProc(_resolved_mcp("known-server"))
    verification = _FakeProc(
        json.dumps(
            {
                "mcp": {
                    "known-server": {"enabled": False},
                    "new-server": {
                        "type": "local",
                        "command": ["mcp-server"],
                        "enabled": True,
                    },
                }
            }
        )
    )
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[discovered, verification],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert all(call.args[0][1:3] == ["debug", "config"] for call in run.call_args_list)
    assert "disable every configured MCP server" in be.last_call_error


@pytest.mark.parametrize(
    ("bad_result", "error_fragment"),
    [
        (subprocess.TimeoutExpired("opencode", 1), "timed out"),
        (OSError("private config path"), "could not be completed"),
        (_FakeProc("", returncode=3), "exited 3"),
        (_FakeProc("not json"), "invalid configuration"),
        (_FakeProc('{"mcp":' + "9" * 5000 + "}"), "invalid configuration"),
        (_FakeProc(json.dumps(["not", "an", "object"])), "invalid configuration"),
        (_FakeProc(json.dumps({"mcp": []})), "invalid MCP configuration"),
        (
            _FakeProc(json.dumps({"mcp": {"server": "not an object"}})),
            "invalid MCP configuration",
        ),
    ],
)
def test_mcp_discovery_failure_stops_before_model_call(bad_result, error_fragment):
    be = OpenCodeCliBackend(opencode_path="opencode", timeout=1)

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[bad_result],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 1
    assert run.call_args.args[0][1:3] == ["debug", "config"]
    assert error_fragment in be.last_call_error
    assert "private config path" not in be.last_call_error


@pytest.mark.parametrize(
    "bad_verification",
    [
        subprocess.TimeoutExpired("opencode", 1),
        _FakeProc("", returncode=4),
        _FakeProc("not json"),
        _FakeProc(json.dumps({"mcp": {"server": []}})),
    ],
)
def test_mcp_verification_failure_stops_before_model_call(bad_verification):
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[_FakeProc(_resolved_mcp("server")), bad_verification],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert all(call.args[0][1:3] == ["debug", "config"] for call in run.call_args_list)
    assert "MCP verification" in be.last_call_error


def test_resolved_config_secrets_are_not_exposed(caplog):
    secret = "resolved-provider-secret-74f92"
    discovered = _FakeProc(
        json.dumps(
            {
                "provider": {"custom": {"options": {"apiKey": secret}}},
                "mcp": {
                    "private-server": {
                        "type": "remote",
                        "url": "https://mcp.invalid",
                        "headers": {"Authorization": secret},
                    }
                },
            }
        )
    )
    verification = _FakeProc(secret, stderr=secret, returncode=5)
    be = OpenCodeCliBackend(opencode_path="opencode")

    with mock.patch(
        "skillopt_sleep.backend.subprocess.run",
        side_effect=[discovered, verification],
    ) as run:
        assert be._call("hello") == ""

    assert run.call_count == 2
    assert secret not in be.last_call_error
    assert secret not in caplog.text
    assert secret not in run.call_args.kwargs["env"]["OPENCODE_CONFIG_CONTENT"]


def test_mcp_checks_run_once_per_cache_miss():
    be = OpenCodeCliBackend(opencode_path="opencode")
    results = _successful_plain_results("server") + _successful_plain_results("server")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=results) as run:
        assert be._cached_call("attempt:first", "first prompt") == "answer"
        assert be._cached_call("attempt:first", "first prompt") == "answer"
        assert be._cached_call("attempt:second", "second prompt") == "answer"

    commands = [call.args[0][1:3] for call in run.call_args_list]
    assert commands == [
        ["debug", "config"],
        ["debug", "config"],
        ["run", "--pure"],
        ["debug", "config"],
        ["debug", "config"],
        ["run", "--pure"],
    ]


def test_failed_call_is_not_cached():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch.object(be, "_call", side_effect=["", "recovered"]) as call:
        assert be._cached_call("attempt:key", "prompt") == ""
        assert be._cached_call("attempt:key", "prompt") == "recovered"
    assert call.call_count == 2


def test_cached_success_clears_stale_call_error():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch.object(be, "_call", return_value="answer") as call:
        assert be._cached_call("attempt:key", "prompt") == "answer"
        be.last_call_error = "unrelated later failure"
        assert be._cached_call("attempt:key", "prompt") == "answer"
    assert call.call_count == 1
    assert be.last_call_error == ""


def test_attempt_with_tools_fails_without_starting_child():
    be = OpenCodeCliBackend(opencode_path="opencode")
    with mock.patch("skillopt_sleep.backend.subprocess.run") as run:
        assert be.attempt_with_tools(mock.Mock(), "skill", "memory", ["search"]) == ("", [])
    run.assert_not_called()
    assert "not supported" in be.last_call_error


def test_get_and_build_backend_route_opencode_path():
    with mock.patch("shutil.which", return_value=None):
        for alias in ("opencode", "opencode_cli", "opencode-cli", "OPENCODE"):
            assert isinstance(get_backend(alias), OpenCodeCliBackend)

        single = build_backend(backend="opencode", opencode_path="custom-opencode")
        assert isinstance(single, OpenCodeCliBackend)
        assert single.opencode_path == "custom-opencode"

        dual = build_backend(
            backend="mock",
            optimizer_backend="opencode",
            target_backend="opencode",
            opencode_path="custom-opencode",
        )
        assert isinstance(dual, DualBackend)
        assert dual.optimizer.opencode_path == "custom-opencode"
        assert dual.target.opencode_path == "custom-opencode"


def test_cli_makes_relative_opencode_path_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    _add_common(parser)
    relative = os.path.join("bin", "opencode")
    args = parser.parse_args(["--backend", "opencode", "--opencode-path", relative])
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)
    cfg = _cfg_from_args(args)
    assert cfg.get("backend") == "opencode"
    assert cfg.get("opencode_path") == os.path.abspath(relative)


def test_cycle_diagnostic_build_forwards_opencode_path():
    cfg = load_config(backend="opencode", opencode_path="custom-opencode")
    with mock.patch("skillopt_sleep.cycle.build_backend", return_value=MockBackend()) as builder:
        cycle._make_model_key(cfg)
    assert builder.call_args.kwargs["opencode_path"] == "custom-opencode"


def test_runtime_cycle_build_forwards_opencode_path(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    cfg = SleepConfig(
        data={
            **DEFAULTS,
            "backend": "opencode",
            "opencode_path": "custom-opencode",
            "projects": "invoked",
            "invoked_project": str(project),
            "state_dir": str(tmp_path / "state"),
            "claude_home": str(tmp_path / "claude-home"),
            "evidence_log": False,
        }
    )

    with mock.patch("skillopt_sleep.cycle.build_backend", return_value=MockBackend()) as builder:
        outcome = cycle.run_sleep_cycle(cfg, seed_tasks=[], dry_run=True)

    assert outcome.report.n_tasks == 0
    assert builder.call_count == 1
    assert builder.call_args.kwargs["opencode_path"] == "custom-opencode"
