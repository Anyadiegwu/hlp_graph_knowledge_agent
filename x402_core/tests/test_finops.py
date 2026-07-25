"""
Run: uv run --package x402_core pytest x402_core/tests/test_finops.py -v

Forces the in-memory ledger path (no REDIS_URL) so this suite never needs
a live Redis instance.
"""
import pytest

from x402_core.exceptions import FinOpsBudgetExceededException
from x402_core.finops import FinOpsGovernor


@pytest.fixture
def governor() -> FinOpsGovernor:
    # redis_url="" forces the in-memory dict path regardless of any
    # REDIS_URL set in the calling shell's environment.
    return FinOpsGovernor(max_usd_per_chain=0.05, redis_url="")


def test_preflight_passes_under_ceiling(governor):
    governor.preflight("session-a", projected_usd=0.01)  # should not raise


def test_preflight_raises_over_ceiling(governor):
    governor.record_spend("session-a", 0.049)
    with pytest.raises(FinOpsBudgetExceededException) as exc_info:
        governor.preflight("session-a", projected_usd=0.01, step_label="tot_thought_branch")
    err = exc_info.value
    assert err.session_id == "session-a"
    assert err.ceiling_usd == 0.05
    assert err.detail["step"] == "tot_thought_branch"


def test_sessions_are_isolated(governor):
    governor.record_spend("session-a", 0.049)
    # session-b has spent nothing, so the same projected cost should pass.
    governor.preflight("session-b", projected_usd=0.01)


def test_record_spend_accumulates(governor):
    total_1 = governor.record_spend("session-c", 0.01)
    total_2 = governor.record_spend("session-c", 0.02)
    assert total_1 == 0.01
    assert total_2 == pytest.approx(0.03)
    assert governor.get_spent("session-c") == pytest.approx(0.03)


def test_tokens_to_usd_conversion(governor):
    usd = governor.tokens_to_usd(1000, usd_per_1k=0.0006)
    assert usd == pytest.approx(0.0006)


def test_velocity_reflects_recent_spend(governor):
    governor.record_spend("session-d", 0.02)
    governor.record_spend("session-d", 0.02)
    velocity = governor.velocity_usd_per_hour("session-d")
    assert velocity > 0


def test_velocity_zero_with_no_spend(governor):
    assert governor.velocity_usd_per_hour("never-spent-session") == 0.0
