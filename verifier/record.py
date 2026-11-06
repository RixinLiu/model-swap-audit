"""read_probes', read_log's and audit_record's checks: the case asked for, its data whole and consistent."""

from .checks import case_problem, result
from .rules import audits


def verify_input(value):
    problem = case_problem(value.case)
    if problem:
        return result(False, problem)
    return result(True, "input is valid")


def probes_problem(output):
    """What is wrong with a case's models, probes and references, or None."""
    if output.models != sorted(set(output.models)):
        return "the known models must be unique and sorted by name"
    probe_ids = [probe.prompt_id for probe in output.probes]
    if len(set(probe_ids)) != len(probe_ids):
        return "a probe id appears twice"
    pairs = [(reference.model, reference.prompt_id) for reference in output.references]
    wanted = [(model, prompt_id) for prompt_id in probe_ids for model in output.models]
    if sorted(pairs) != sorted(wanted):
        return "every known model must have exactly one reference for every probe"
    return None


def log_problem(output):
    call_ids = [call.call_id for call in output.calls]
    if len(set(call_ids)) != len(call_ids):
        return "a call id appears twice in the log"
    return None


def record_problem(output):
    """What is wrong with a whole case, its labels included, or None. Labels must agree with the rules."""
    problem = probes_problem(output) or log_problem(output)
    if problem:
        return problem
    if [label.call_id for label in output.labels] != [call.call_id for call in output.calls]:
        return "the labels do not follow the log's calls one for one"
    for label, (call_id, verdict, matched) in zip(output.labels, audits(output)):
        if (label.verdict, label.matched_models) != (verdict, matched):
            return f"the generator's label for {call_id} disagrees with the rules"
    return None


def checked(input_value, output_value, problem):
    if output_value.case != input_value.case:
        return result(False, f"asked for case {input_value.case}, got case {output_value.case}")
    if problem:
        return result(False, problem)
    return result(True, "the case is well formed")


def verify_probes_output(input_value, output_value, workspace):
    return checked(input_value, output_value, probes_problem(output_value))


def verify_log_output(input_value, output_value, workspace):
    return checked(input_value, output_value, log_problem(output_value))


def verify_record_output(input_value, output_value, workspace):
    return checked(input_value, output_value, record_problem(output_value))
