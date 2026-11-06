"""The audit rules, applied by the verifier on its own to a case's data: no generator knowledge, only strings."""

from contract import Verdict


def reference_table(references):
    """prompt id -> model -> its reference output."""
    table = {}
    for reference in references:
        table.setdefault(reference.prompt_id, {})[reference.model] = reference.output
    return table


def audit(call, probe_ids, table):
    """``call``'s (verdict, matched_models) under the rules."""
    if call.temperature != 0 or call.prompt_id not in probe_ids:
        return Verdict.UNVERIFIABLE, []
    matched = sorted(model for model, text in table.get(call.prompt_id, {}).items() if text == call.output)
    if not matched:
        return Verdict.UNKNOWN, []
    if call.claimed_model in matched:
        return Verdict.GENUINE, matched
    return Verdict.SWAPPED, matched


def audits(record):
    """Every call's (call_id, verdict, matched_models), in log order. ``record``: a contract RecordOutput."""
    probe_ids = {probe.prompt_id for probe in record.probes}
    table = reference_table(record.references)
    return [(call.call_id, *audit(call, probe_ids, table)) for call in record.calls]
