"""A failed process binding must explain itself, not hide behind a generic id.

At approval-PROPOSE time tool_approvals resolves the sealed process operation
best-effort. When that resolution fails (e.g. the workspace was never sealed as
a launch scope) the pending must record the real reason, and replay must report
THAT reason instead of the generic "no sealed process/job identity".
"""
import asyncio
from collections import namedtuple

from src.agent_runtime.authority import OperationGrant, RequestAuthority
from src.tool_approvals import ToolApprovalStore
from src.tool_capabilities import capabilities_for_action
from src.tool_execution import NO_TOOL_SECURITY_CONTEXT, execute_tool_block

ToolBlock = namedtuple("ToolBlock", ["tool_type", "content"])


def _pending_without_launch_scope(store):
    # A bash grant but NO launch scope: resolve_process_operation cannot seal a
    # process operation, which is exactly the swallowed failure under test.
    authority = RequestAuthority("req-1", "alice", "session-1", "",
                                 (OperationGrant("bash"),), launch_scopes=())
    return store.create(
        owner="Alice", session_id="session-1", origin_run_id="run-1",
        tool_name="bash", content="printf exact", workspace=None,
        external_untrusted_context_seen=True,
        capabilities=capabilities_for_action("bash", "printf exact"),
        request_authority=authority,
    )


def test_failed_process_binding_records_its_reason_at_propose():
    store = ToolApprovalStore()
    pending = _pending_without_launch_scope(store)
    assert pending.process_operation is None
    assert pending.process_operation_denial
    assert "launch scope" in pending.process_operation_denial


def test_replay_reports_the_recorded_reason_not_a_generic_identity_error():
    store = ToolApprovalStore()
    pending = _pending_without_launch_scope(store)
    grant = store.consume(pending.approval_id, decision="approve",
                          owner="alice", session_id="session-1")
    assert grant is not None

    async def _run():
        return await execute_tool_block(
            ToolBlock("bash", "printf exact"), session_id="session-1", owner="alice",
            workspace=None, security_context=NO_TOOL_SECURITY_CONTEXT,
            exact_approval=grant, request_authority=pending.request_authority,
        )

    description, result = asyncio.run(_run())
    assert "BLOCKED" in description
    assert result["failure_kind"] == "resource_identity_denied"
    assert "launch scope" in result["error"]
    assert "no sealed process/job identity" not in result["error"]
