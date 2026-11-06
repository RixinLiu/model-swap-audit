"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    READ_PROBES = "read_probes"
    READ_LOG = "read_log"
    COMPARE_OUTPUT = "compare_output"
    AUDIT_RECORD = "audit_record"


class Task(Enum):
    AUDIT_CALLS = "audit-calls"


class Verdict(Enum):
    GENUINE = "genuine"
    SWAPPED = "swapped"
    UNKNOWN = "unknown"
    UNVERIFIABLE = "unverifiable"


# How many built-in cases there are: a case is numbered 1 to CASES.
CASES = 6

CASE = "The case's number, 1 to 6"
VERDICT = (
    "unverifiable: temperature is not 0 or the prompt is not a probe; genuine: the claimed model is among"
    " matched_models; swapped: matched_models is not empty and lacks the claimed model; unknown: no known"
    " model matches"
)
MATCHED = (
    "The known models whose reference output for the call's prompt equals its output exactly, sorted by name;"
    " empty for an unverifiable call"
)


@dataclass
class Probe(Output):
    prompt_id: Annotated[str, "The probe's id, which calls name"]
    prompt: Annotated[str, "The probe prompt's text"]


@dataclass
class Reference(Output):
    model: Annotated[str, "A known model"]
    prompt_id: Annotated[str, "A probe's id"]
    output: Annotated[str, "What the model answers to the probe at temperature 0, exactly"]


@dataclass
class Call(Output):
    call_id: Annotated[str, "The call's id in the log"]
    claimed_model: Annotated[str, "The model the API said answered the call"]
    prompt_id: Annotated[str, "The prompt the call sent: a probe's id, or another prompt's"]
    temperature: Annotated[float, "The sampling temperature the call asked for"]
    output: Annotated[str, "What the API returned, exactly"]


@dataclass
class Label(Output):
    call_id: Annotated[str, "The call's id in the log"]
    source_model: Annotated[str, "The model that really answered the call: a known model or one outside the known set"]
    verdict: Annotated[Verdict, VERDICT]
    matched_models: Annotated[list[str], MATCHED]


@dataclass
class Match(Output):
    model: Annotated[str, "A known model"]
    equal: Annotated[bool, "Whether the call's output equals the model's reference for the call's prompt exactly"]


@dataclass
class Audit(Output):
    call_id: Annotated[str, "The call's id in the log"]
    verdict: Annotated[Verdict, VERDICT]
    matched_models: Annotated[list[str], MATCHED]


@dataclass
class CaseInput(Output):
    case: Annotated[int, CASE]


@dataclass
class ProbesOutput(Output):
    case: Annotated[int, CASE]
    models: Annotated[list[str], "The known models, sorted by name"]
    probes: Annotated[list[Probe], "The probe prompts"]
    references: Annotated[list[Reference], "Each known model's output for each probe at temperature 0"]


@dataclass
class LogOutput(Output):
    case: Annotated[int, CASE]
    calls: Annotated[list[Call], "The API's calls, in the log's order"]


@dataclass
class CompareInput(Output):
    case: Annotated[int, CASE]
    call_id: Annotated[str, "A call's id in the log"]
    models: Annotated[list[str], "Known models to compare the call's output with"]


@dataclass
class CompareOutput(Output):
    call_id: Annotated[str, "The call's id in the log"]
    prompt_id: Annotated[str, "The probe the call sent"]
    matches: Annotated[list[Match], "For each model asked about, in order, whether its reference equals the output"]


@dataclass
class RecordOutput(Output):
    case: Annotated[int, CASE]
    models: Annotated[list[str], "The known models, sorted by name"]
    probes: Annotated[list[Probe], "The probe prompts"]
    references: Annotated[list[Reference], "Each known model's output for each probe at temperature 0"]
    calls: Annotated[list[Call], "The API's calls, in the log's order"]
    labels: Annotated[list[Label], "The data generator's label for each call, in the log's order"]


@dataclass
class AuditOutput(Output):
    audits: Annotated[list[Audit], "One audit per call, in the log's order"]


TOOLS = {
    Tool.READ_PROBES: (CaseInput, ProbesOutput),
    Tool.READ_LOG: (CaseInput, LogOutput),
    Tool.COMPARE_OUTPUT: (CompareInput, CompareOutput),
    Tool.AUDIT_RECORD: (CaseInput, RecordOutput),
}
TASKS = {Task.AUDIT_CALLS: (CaseInput, AuditOutput)}
