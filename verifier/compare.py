"""compare_output's checks: one answer per model asked about, for the call asked about."""

from .checks import case_problem, result


def verify_input(value):
    problem = case_problem(value.case)
    if problem:
        return result(False, problem)
    if not value.call_id.strip():
        return result(False, "call_id must not be blank")
    if not value.models:
        return result(False, "name at least one known model")
    if len(set(value.models)) != len(value.models):
        return result(False, "a model is named twice")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.call_id != input_value.call_id:
        return result(False, f"asked about {input_value.call_id}, got {output_value.call_id}")
    if [match.model for match in output_value.matches] != input_value.models:
        return result(False, "the matches must answer the models asked about, in order")
    return result(True, "compared", equal=sum(match.equal for match in output_value.matches))
