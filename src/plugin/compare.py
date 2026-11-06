"""compare_output: whether a call's output equals known models' references for its prompt, exactly."""

from __future__ import annotations

from pathlib import Path

from contract import CompareInput, CompareOutput, Match
from plugin import cases


def compare_output(given: CompareInput, workspace: Path) -> CompareOutput:
    record = cases.audit_record(given.case)
    call = next((call for call in record.calls if call.call_id == given.call_id), None)
    if call is None:
        raise ValueError(f"case {given.case}'s log has no call {given.call_id!r}")
    references = {(ref.model, ref.prompt_id): ref.output for ref in record.references}
    if not any(prompt_id == call.prompt_id for _, prompt_id in references):
        raise ValueError(f"{call.prompt_id!r} is not a probe, so no model has a reference for it")
    if not given.models:
        raise ValueError("name at least one known model")
    for model in given.models:
        if model not in record.models:
            raise ValueError(f"{model!r} is not a known model; the known models are {', '.join(record.models)}")
    return CompareOutput(
        call_id=call.call_id,
        prompt_id=call.prompt_id,
        matches=[Match(model=model, equal=references[model, call.prompt_id] == call.output) for model in given.models],
    )
