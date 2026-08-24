"""Tests for attest.provenance.protocol: the validation-protocol descriptor and its own id."""

from __future__ import annotations

import pytest

from attest.provenance.protocol import (
    AdjudicationProtocol,
    AuditDesign,
    ProtocolError,
    ReportingSpec,
    SentinelPolicy,
    ValidationProtocol,
    compute_protocol_id,
)


def _protocol() -> ValidationProtocol:
    return ValidationProtocol(
        audit_design=AuditDesign(
            stratify_by="track", audit_size_policy="n=600", confidence_level=0.95
        ),
        adjudication_protocol=AdjudicationProtocol(protocol_version="1", description="dual review"),
        sentinel_policy=SentinelPolicy(hard_trigger_crossings=2, advisory_alpha_threshold=0.8),
        reporting_spec=ReportingSpec(validation_record_schema_version="1.1", notes="PRISMA 2020"),
    )


def test_same_content_yields_same_protocol_id() -> None:
    assert compute_protocol_id(_protocol()) == compute_protocol_id(_protocol())


def test_different_audit_design_changes_protocol_id() -> None:
    base = _protocol()
    changed = ValidationProtocol(
        audit_design=AuditDesign(stratify_by="confidence"),
        adjudication_protocol=base.adjudication_protocol,
        sentinel_policy=base.sentinel_policy,
        reporting_spec=base.reporting_spec,
    )

    assert compute_protocol_id(base) != compute_protocol_id(changed)


def test_protocol_to_dict_and_from_dict_round_trip() -> None:
    protocol = _protocol()

    restored = ValidationProtocol.from_dict(protocol.to_dict())

    assert restored == protocol
    assert compute_protocol_id(restored) == compute_protocol_id(protocol)


def test_audit_design_rejects_unknown_stratification() -> None:
    with pytest.raises(ProtocolError):
        AuditDesign(stratify_by="topic")


def test_audit_design_rejects_confidence_level_outside_open_unit_interval() -> None:
    with pytest.raises(ProtocolError):
        AuditDesign(confidence_level=1.0)


def test_sentinel_policy_rejects_non_positive_crossings() -> None:
    with pytest.raises(ProtocolError):
        SentinelPolicy(hard_trigger_crossings=0)


def test_sentinel_policy_rejects_alpha_threshold_outside_unit_interval() -> None:
    with pytest.raises(ProtocolError):
        SentinelPolicy(advisory_alpha_threshold=1.5)


def test_default_protocol_constructs_with_no_arguments() -> None:
    protocol = ValidationProtocol()

    assert protocol.audit_design.stratify_by == "none"
    assert protocol.sentinel_policy.hard_trigger_crossings == 2
