"""The audit-calls task's check: one audit per call, in log order, each equal to the generator's label."""

from .checks import result
from .record import record_problem


def audit_problem(audit, label):
    """What is wrong with one call's audit, naming the field, or None."""
    if audit.verdict != label.verdict:
        return f"{audit.call_id}: verdict is {audit.verdict.value}, expected {label.verdict.value}"
    if audit.matched_models != label.matched_models:
        if sorted(audit.matched_models) == label.matched_models:
            return f"{audit.call_id}: matched_models {audit.matched_models} are not sorted by name"
        return f"{audit.call_id}: matched_models is {audit.matched_models}, expected {label.matched_models}"
    return None


def verify_task(arguments, output, reference, workspace):
    """``reference``: the contract RecordOutput of the case asked for, pinned before the agent ran."""
    if reference is None or reference.case != arguments.case:
        return result(False, f"the pinned reference is not case {arguments.case}")
    problem = record_problem(reference)
    if problem:
        return result(False, f"the pinned reference cannot be used: {problem}")
    labels = {label.call_id: label for label in reference.labels}
    order = [label.call_id for label in reference.labels]
    submitted = [audit.call_id for audit in output.audits]
    # The share of positions whose audit is the right call's, fully right, over the calls or the audits if more.
    right = sum(
        1
        for audit, call_id in zip(output.audits, order)
        if audit.call_id == call_id and audit_problem(audit, labels[call_id]) is None
    )
    total = max(len(order), len(submitted))
    score = 1.0 if total == 0 else right / total
    for call_id in submitted:
        if call_id not in labels:
            return result(False, f"{call_id} is not a call in the log", score=score)
    duplicated = [call_id for call_id in order if submitted.count(call_id) > 1]
    if duplicated:
        return result(False, f"{duplicated[0]} is audited more than once", score=score)
    missing = [call_id for call_id in order if call_id not in submitted]
    if missing:
        return result(False, f"{missing[0]} is not audited", score=score, missing=missing)
    if submitted != order:
        position = next(i for i, (got, want) in enumerate(zip(submitted, order)) if got != want)
        return result(
            False, f"audits are not in the log's order: audit {position} is {submitted[position]}, expected"
            f" {order[position]}", score=score,
        )
    for audit in output.audits:
        problem = audit_problem(audit, labels[audit.call_id])
        if problem:
            return result(False, problem, score=score)
    return result(True, f"all {len(order)} calls are audited right", calls=len(order))
