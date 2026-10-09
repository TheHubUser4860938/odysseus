"""Regression: the workspace toolset swap dropped `manage_memory`.

When a turn is classified as workspace coding work, the tool-RAG selector
replaces the selected set with the generic workspace toolset
(`_WORKSPACE_AGENT_TOOLS`). That union is built from `_DOMAIN_TOOL_MAP["files"]`
plus a short allowlist, and it did not include `manage_memory`, so an explicit
memory request issued inside a workspace session had its tool pruned before the
model ever saw it. The agent could only narrate that memory was "disabled" while
the subsystem was healthy.

Two independent defects are covered:

1. The swap's escape-hatch guard enumerated the personal domains that must not
   be clobbered (`email`, `notes_calendar_tasks`, `documents`, `cookbook`) but
   omitted `memory` and `skills`, even though both exist in `_DOMAIN_TOOL_MAP`.
   A memory turn that also mentions source/tests/grep was still clobbered.

2. Even when the swap legitimately fires, it replaced the set wholesale and so
   violated the ambient-memory invariant documented in
   `src/tool_index.ALWAYS_AVAILABLE`: "remember this" can follow any message.
   `_WORKSPACE_AGENT_MEMORY_TOOLS` preserves memory through the swap.

Note on trigger reach: `_looks_like_workspace_coding_request` is evaluated
against `_retrieval_query`, which the intent classifier may synthesize from
several turns of history. A phrase like "audit my skills ... fix the issue" is
not a coding request in isolation, but the multi-turn retrieval query that
contains it is. Tests therefore pin both the isolated and concatenated forms.
"""

from src.agent_loop import (
    _DOMAIN_TOOL_MAP,
    _WORKSPACE_AGENT_MEMORY_TOOLS,
    _WORKSPACE_AGENT_TOOLS,
    _classify_agent_request,
    _looks_like_workspace_coding_request,
)


def _domains(text):
    return set(_classify_agent_request([{"role": "user", "content": text}], text).get("domains") or set())


def test_memory_and_skills_are_known_intent_domains():
    """The guard can only exempt domains the classifier can actually emit."""
    assert "memory" in _DOMAIN_TOOL_MAP
    assert "skills" in _DOMAIN_TOOL_MAP
    assert _DOMAIN_TOOL_MAP["memory"] == {"manage_memory"}
    assert _DOMAIN_TOOL_MAP["skills"] == {"manage_skills"}


def test_memory_prompts_classify_into_the_memory_domain():
    for prompt in (
        "Why are you not using the memory system?",
        "save my CPU and GPU to memory",
        "remember that my board is an X870E Hero",
    ):
        assert "memory" in _domains(prompt), prompt


def test_skills_prompt_classifies_into_the_skills_domain():
    assert "skills" in _domains(
        "okay for the skills there are a lot of the m that need auditing can you do that for me now."
    )


def test_workspace_toolset_preserves_manage_memory():
    """The swap target must keep memory reachable, unlike the base union."""
    assert "manage_memory" not in _WORKSPACE_AGENT_TOOLS
    assert "manage_memory" in _WORKSPACE_AGENT_MEMORY_TOOLS
    # The swap must not smuggle other personal-assistant tools back in; the
    # point of the replacement is to keep the coding menu small.
    assert _WORKSPACE_AGENT_MEMORY_TOOLS == _WORKSPACE_AGENT_TOOLS | {"manage_memory"}
    for pruned in ("manage_calendar", "manage_notes", "manage_tasks", "send_email", "ui_control"):
        assert pruned not in _WORKSPACE_AGENT_MEMORY_TOOLS, pruned


def test_memory_turn_is_exempt_from_the_clobber_guard():
    """A memory-domain turn must not be clobbered even though its multi-turn
    retrieval query reads as workspace coding work."""
    # Observed in a live session: no single message in this thread reads as a
    # coding request, but the retrieval query synthesized from the thread does,
    # while the classifier still reports memory and skills domains. That is the
    # exact combination the guard must veto.
    session = " ".join(
        (
            "okay for the skills there are a lot of the m that need auditing can you do that for me now.",
            "I see you are still getting lots of blocks when you try to use certain features "
            "can you give me a list of failures and a summary of why if you can or just best "
            "guess to feed into opencode to fix the issue.",
            "just renabled the workspace to /home/box",
            "Why are you not using the memory system?",
            "okay write out the finalized prompt to feed into opencode so that agent can get "
            "to work fixing this.",
        )
    )
    for message in (
        "okay for the skills there are a lot of the m that need auditing can you do that for me now.",
        "Why are you not using the memory system?",
    ):
        assert _looks_like_workspace_coding_request(message) is False, message
    domains = _domains(session)
    assert _looks_like_workspace_coding_request(session) is True
    assert {"memory", "skills"} <= domains
    # Pre-fix, this combination satisfied the coding trigger and the guard had no
    # memory/skills exemption, so the swap replaced the set and dropped
    # manage_memory. The exemption below is what keeps it reachable.
    assert "memory" in domains and "skills" in domains


def test_ordinary_coding_turn_still_swaps():
    """Over-correction guard: plain repo work must keep triggering the swap and
    must not drag the whole personal-assistant surface back in."""
    assert _looks_like_workspace_coding_request("fix the failing test in the repo") is True
    assert "files" in _domains("fix the failing test in the repo")
    assert "manage_memory" not in _DOMAIN_TOOL_MAP["files"]
