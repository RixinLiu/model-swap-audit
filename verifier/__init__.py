"""The deterministic verifier: each tool's checks live in its own module, on contract instances."""

from contract import Task, Tool

from . import answer, compare, record

# tool -> (verify_input(value), verify_output(input_value, output_value, workspace))
TOOLS = {
    Tool.READ_PROBES: (record.verify_input, record.verify_probes_output),
    Tool.READ_LOG: (record.verify_input, record.verify_log_output),
    Tool.COMPARE_OUTPUT: (compare.verify_input, compare.verify_output),
    Tool.AUDIT_RECORD: (record.verify_input, record.verify_record_output),
}
# task -> its output's check, verify_task(arguments, output, reference, workspace)
TASKS = {Task.AUDIT_CALLS: answer.verify_task}


def verify_input(tool, value):
    return TOOLS[Tool(tool)][0](value)


def verify_output(tool, input_value, output_value, workspace):
    return TOOLS[Tool(tool)][1](input_value, output_value, workspace)


def verify_task(task, arguments, output, reference, workspace):
    return TASKS[Task(task)](arguments, output, reference, workspace)
