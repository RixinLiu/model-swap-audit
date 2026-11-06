"""Checks shared by the tool verifiers. Pure functions of their arguments."""

from contract import CASES


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def case_problem(case):
    if not 1 <= case <= CASES:
        return f"case must be 1 to {CASES}, not {case}"
    return None
