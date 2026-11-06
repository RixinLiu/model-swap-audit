"""read_probes and read_log show the agent a case; audit_record, the task's reference, adds the generator's labels."""

from __future__ import annotations

from pathlib import Path

from contract import CaseInput, LogOutput, ProbesOutput, RecordOutput
from plugin import cases


def audit_record(given: CaseInput, workspace: Path) -> RecordOutput:
    return cases.audit_record(given.case)


def read_probes(given: CaseInput, workspace: Path) -> ProbesOutput:
    record = cases.audit_record(given.case)
    return ProbesOutput(case=record.case, models=record.models, probes=record.probes, references=record.references)


def read_log(given: CaseInput, workspace: Path) -> LogOutput:
    record = cases.audit_record(given.case)
    return LogOutput(case=record.case, calls=record.calls)
