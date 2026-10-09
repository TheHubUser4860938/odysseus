"""Regression tests for polling endpoints and foreground task interruption."""

import asyncio
import importlib


def _reload_gate():
    import src.interactive_gate as ig

    importlib.reload(ig)
    return ig


def test_tasks_runs_recent_is_passive():
    ig = _reload_gate()

    assert not ig.should_track_interactive_request(
        "/api/tasks/runs/recent", "GET"
    )


def test_tasks_runs_recent_does_not_affect_other_task_paths(monkeypatch):
    # LOCAL FORK: the foreground gate now defaults off for single-user local
    # installs (upstream default was on); pin the gate-enabled premise.
    monkeypatch.setenv("BACKGROUND_TASK_FOREGROUND_GATE", "true")
    ig = _reload_gate()

    # A neighboring mutating path must remain interactive.
    assert ig.should_track_interactive_request(
        "/api/tasks/runs/recent/something", "POST"
    )


def test_heartbeat_does_not_stop_background_tasks_when_gate_disabled(monkeypatch):
    ig = _reload_gate()
    monkeypatch.setenv("BACKGROUND_TASK_FOREGROUND_GATE", "false")

    stop_calls = []

    async def fake_stop_background_tasks_for_foreground(*, reason):
        stop_calls.append(reason)

    result = asyncio.run(
        ig.maybe_stop_background_tasks_for_heartbeat(
            fake_stop_background_tasks_for_foreground
        )
    )

    assert result is False
    assert stop_calls == []


def test_heartbeat_stops_background_tasks_when_gate_enabled(monkeypatch):
    # LOCAL FORK: gate-enabled premise pinned explicitly (upstream defaulted
    # the gate on; this build defaults it off for single-user installs).
    monkeypatch.setenv("BACKGROUND_TASK_FOREGROUND_GATE", "true")
    ig = _reload_gate()

    stop_calls = []

    async def fake_stop_background_tasks_for_foreground(*, reason):
        stop_calls.append(reason)

    result = asyncio.run(
        ig.maybe_stop_background_tasks_for_heartbeat(
            fake_stop_background_tasks_for_foreground
        )
    )

    assert result is True
    assert stop_calls == ["browser heartbeat"]
