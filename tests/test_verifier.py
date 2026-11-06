from __future__ import annotations

import difflib
from dataclasses import replace
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import CASES, TASKS, TOOLS, AuditOutput, RecordOutput, Task, Tool, Verdict
from plugin import cases

TASK = Task.AUDIT_CALLS.value
GENUINE, SWAPPED, UNKNOWN, UNVERIFIABLE = (Verdict.GENUINE, Verdict.SWAPPED, Verdict.UNKNOWN, Verdict.UNVERIFIABLE)


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verify_task(case: int, output: dict, reference: RecordOutput | None, workspace: Path) -> dict:
    given, made = TASKS[Task.AUDIT_CALLS]
    pinned = None if reference is None else build_output(plain_output(reference), RecordOutput)
    return verifier.verify_task(TASK, build_output({"case": case}, given), build_output(output, made), pinned, str(workspace))


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def verify_output(tool: Tool, given: dict, made: dict) -> dict:
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(tool.value, build_output(given, given_class), build_output(made, made_class), ".")


def answer(audits: list[tuple[str, Verdict, list[str]]]) -> dict:
    return {
        "audits": [
            {"call_id": call_id, "verdict": verdict.value, "matched_models": matched} for call_id, verdict, matched in audits
        ]
    }


def right(record: RecordOutput) -> list[tuple[str, Verdict, list[str]]]:
    """The generator's labels, which the verifier holds the agent to."""
    return [(label.call_id, label.verdict, label.matched_models) for label in record.labels]


def by_rules(record: RecordOutput, *, ignore_temperature=False, audit_unprobed=False, closest=False):
    """The audit rules with one of them changed, as an agent that misreads it would apply them."""
    probes = {probe.prompt_id for probe in record.probes}
    table: dict[str, dict[str, str]] = {}
    for reference in record.references:
        table.setdefault(reference.prompt_id, {})[reference.model] = reference.output
    audits = []
    for call in record.calls:
        hot = call.temperature != 0 and not ignore_temperature
        if hot or (call.prompt_id not in probes and not audit_unprobed):
            audits.append((call.call_id, UNVERIFIABLE, []))
            continue
        outputs = table.get(call.prompt_id, {})
        matched = sorted(model for model, text in outputs.items() if text == call.output)
        if not matched and closest and outputs:
            matched = [max(outputs, key=lambda model: difflib.SequenceMatcher(None, outputs[model], call.output).ratio())]
        if not matched:
            audits.append((call.call_id, UNKNOWN, []))
        else:
            audits.append((call.call_id, GENUINE if call.claimed_model in matched else SWAPPED, matched))
    return audits


def mutants(record: RecordOutput) -> dict[str, list[tuple[str, Verdict, list[str]]]]:
    """Wrong answers an agent could submit, each wrong in one way; only those that differ from the right one."""
    good = right(record)
    known = set(record.models)
    claimed = {call.call_id: call.claimed_model for call in record.calls}
    found = {
        "closest model instead of unknown": by_rules(record, closest=True),
        "temperature ignored": by_rules(record, ignore_temperature=True),
        "unprobed prompt audited": by_rules(record, audit_unprobed=True),
        "shared reference reported as one model": [
            (call_id, verdict, [claimed[call_id] if claimed[call_id] in matched else matched[0]] if matched else [])
            for call_id, verdict, matched in good
        ],
        "matched_models unsorted": [(call_id, verdict, matched[::-1]) for call_id, verdict, matched in good],
        "claimed model ignored": [
            (call_id, GENUINE if verdict is SWAPPED else verdict, matched) for call_id, verdict, matched in good
        ],
        "genuine taken for swapped": [
            (call_id, SWAPPED if verdict is GENUINE else verdict, matched) for call_id, verdict, matched in good
        ],
        "unlisted claimed model called unknown": [
            (call_id, UNKNOWN if claimed[call_id] not in known and verdict is SWAPPED else verdict, matched)
            for call_id, verdict, matched in good
        ],
        "unknown call naming a model": [
            (call_id, verdict, [record.models[0]] if verdict is UNKNOWN else matched) for call_id, verdict, matched in good
        ],
        "a call missing": good[:-1],
        "a call duplicated": [good[0], *good],
        "calls reordered": [good[1], good[0], *good[2:]],
        "a call not in the log": [*good, ("c99", UNKNOWN, [])],
        "nothing submitted": [],
    }
    return {name: wrong for name, wrong in found.items() if wrong != good}


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_approves_the_right_audit_of_every_case(number: int, tmp_path: Path) -> None:
    record = cases.audit_record(number)
    verdict = verify_task(number, answer(right(record)), record, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0, verdict["message"]


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_refuses_every_wrong_audit(number: int, tmp_path: Path) -> None:
    record = cases.audit_record(number)
    for name, wrong in mutants(record).items():
        verdict = verify_task(number, answer(wrong), record, tmp_path)
        assert verdict["valid"] is False, f"case {number}: {name} was approved"
        assert 0.0 <= verdict["score"] < 1.0, f"case {number}: {name} scored {verdict['score']}"


def test_every_mistake_shows_in_some_case() -> None:
    fired = set()
    for number in range(1, CASES + 1):
        fired |= set(mutants(cases.audit_record(number)))
    expected = {
        "closest model instead of unknown",
        "temperature ignored",
        "unprobed prompt audited",
        "shared reference reported as one model",
        "matched_models unsorted",
        "claimed model ignored",
        "genuine taken for swapped",
        "unlisted claimed model called unknown",
        "unknown call naming a model",
        "a call missing",
        "a call duplicated",
        "calls reordered",
        "a call not in the log",
        "nothing submitted",
    }
    assert fired == expected


def test_case_six_holds_every_mistake() -> None:
    assert len(mutants(cases.audit_record(6))) == 14


def test_each_wrong_audit_says_what_is_wrong(tmp_path: Path) -> None:
    record = cases.audit_record(4)
    wrong = mutants(record)

    def message(name: str) -> str:
        return verify_task(4, answer(wrong[name]), record, tmp_path)["message"]

    assert message("matched_models unsorted") == "c01: matched_models ['cobalt-34b', 'atlas-7b'] are not sorted by name"
    assert message("shared reference reported as one model").startswith("c01: matched_models is ['atlas-7b']")
    assert message("a call missing") == "c06 is not audited"
    assert message("a call duplicated") == "c01 is audited more than once"
    assert message("calls reordered") == "audits are not in the log's order: audit 0 is c02, expected c01"
    assert message("a call not in the log") == "c99 is not a call in the log"
    record = cases.audit_record(2)
    closest = verify_task(2, answer(mutants(record)["closest model instead of unknown"]), record, tmp_path)
    assert closest["message"] == "c01: verdict is genuine, expected unknown"
    assert closest["score"] == pytest.approx(2 / 6)


def test_refuses_a_reference_whose_labels_break_the_rules(tmp_path: Path) -> None:
    record = cases.audit_record(1)
    tampered = replace(record, labels=[replace(record.labels[0], verdict=SWAPPED), *record.labels[1:]])
    verdict = verify_task(1, answer(right(tampered)), tampered, tmp_path)
    assert verdict["valid"] is False and "disagrees with the rules" in verdict["message"]
    assert verify_task(1, answer(right(record)), None, tmp_path)["valid"] is False
    assert verify_task(2, answer(right(record)), record, tmp_path)["valid"] is False


@pytest.mark.parametrize(
    "output",
    [
        {"audits": [{"call_id": "c01", "verdict": "maybe", "matched_models": []}]},
        {"audits": [{"call_id": "c01", "verdict": "genuine"}]},
        {"audits": [{"call_id": "c01", "verdict": "genuine", "matched_models": "atlas-7b"}]},
        {"audits": [{"call_id": "c01", "verdict": "genuine", "matched_models": [], "model": "atlas-7b"}]},
        {"audits": {"c01": "genuine"}},
        {},
    ],
)
def test_the_contract_refuses_a_malformed_output_before_the_verifier(output: dict) -> None:
    with pytest.raises(ValueError):
        build_output(output, AuditOutput)


def test_checks_what_the_types_cannot_say() -> None:
    assert verify_input(Tool.READ_PROBES, {"case": 1})["valid"] is True
    assert verify_input(Tool.READ_LOG, {"case": 0})["valid"] is False
    assert verify_input(Tool.AUDIT_RECORD, {"case": CASES + 1})["valid"] is False
    asked = {"case": 1, "call_id": "c01", "models": ["atlas-7b", "borealis-13b"]}
    assert verify_input(Tool.COMPARE_OUTPUT, asked)["valid"] is True
    assert verify_input(Tool.COMPARE_OUTPUT, {**asked, "models": []})["valid"] is False
    assert verify_input(Tool.COMPARE_OUTPUT, {**asked, "models": ["atlas-7b", "atlas-7b"]})["valid"] is False
    assert verify_input(Tool.COMPARE_OUTPUT, {**asked, "call_id": " "})["valid"] is False
    matches = [{"model": "atlas-7b", "equal": True}, {"model": "borealis-13b", "equal": False}]
    compared = {"call_id": "c01", "prompt_id": "p1", "matches": matches}
    assert verify_output(Tool.COMPARE_OUTPUT, asked, compared)["valid"] is True
    assert verify_output(Tool.COMPARE_OUTPUT, asked, {**compared, "matches": matches[::-1]})["valid"] is False
    assert verify_output(Tool.COMPARE_OUTPUT, asked, {**compared, "call_id": "c02"})["valid"] is False


def test_checks_the_case_data() -> None:
    for number in range(1, CASES + 1):
        record = plain_output(cases.audit_record(number))
        assert verify_output(Tool.AUDIT_RECORD, {"case": number}, record)["valid"] is True
        probes = {key: record[key] for key in ("case", "models", "probes", "references")}
        assert verify_output(Tool.READ_PROBES, {"case": number}, probes)["valid"] is True
        log = {"case": number, "calls": record["calls"]}
        assert verify_output(Tool.READ_LOG, {"case": number}, log)["valid"] is True
    record = plain_output(cases.audit_record(1))
    assert verify_output(Tool.AUDIT_RECORD, {"case": 2}, record)["valid"] is False
    assert verify_output(Tool.AUDIT_RECORD, {"case": 1}, {**record, "references": record["references"][1:]})[
        "valid"
    ] is False
    assert verify_output(Tool.AUDIT_RECORD, {"case": 1}, {**record, "labels": record["labels"][::-1]})["valid"] is False
    calls = [*record["calls"], record["calls"][0]]
    assert verify_output(Tool.READ_LOG, {"case": 1}, {"case": 1, "calls": calls})["valid"] is False
    models = {**record, "models": record["models"][::-1]}
    assert verify_output(Tool.AUDIT_RECORD, {"case": 1}, models)["valid"] is False
