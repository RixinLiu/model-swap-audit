from __future__ import annotations

import random
from pathlib import Path

import pytest

from contract import CASES, CaseInput, CompareInput, Verdict
from plugin import audit_record, cases, compare_output, read_log, read_probes
from plugin.cases import KNOWN, MUTATIONS, PROBES, REFERENCES, UNLISTED, CaseSpec, build_case, near_miss, random_calls
from verifier.rules import audits


def labels(record) -> list[tuple[str, Verdict, list[str]]]:
    return [(label.call_id, label.verdict, label.matched_models) for label in record.labels]


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_the_generator_labels_follow_the_rules_on_every_case(number: int) -> None:
    record = cases.audit_record(number)
    assert labels(record) == audits(record)


def random_shared(rng: random.Random) -> dict[str, tuple[str, ...]]:
    """Some probes on which two or three known models answer alike, whose references differ elsewhere."""
    shared = {}
    for prompt_id in sorted(PROBES):
        if rng.random() < 0.3:
            shared[prompt_id] = tuple(rng.sample(KNOWN, rng.choice((2, 3))))
    return shared


@pytest.mark.parametrize("seed", range(400))
def test_the_generator_labels_follow_the_rules_on_random_logs(seed: int) -> None:
    rng = random.Random(seed)
    shared = random_shared(rng)
    record = build_case(0, CaseSpec(calls=random_calls(rng, 25, shared), shared=shared, tests="random"))
    assert labels(record) == audits(record)


def test_cases_are_the_same_on_every_run_and_cover_what_they_say() -> None:
    for number in range(1, CASES + 1):
        assert cases.audit_record(number) == cases.audit_record(number)
    with pytest.raises(ValueError):
        cases.audit_record(CASES + 1)

    def verdicts(number: int) -> list[Verdict]:
        return [label.verdict for label in cases.audit_record(number).labels]

    assert verdicts(1).count(Verdict.SWAPPED) == 1
    assert verdicts(2).count(Verdict.UNKNOWN) == 4
    hot = [call for call in cases.audit_record(3).calls if call.temperature != 0]
    references = {reference.output for reference in cases.audit_record(3).references}
    assert any(call.output in references for call in hot)
    assert any(len(label.matched_models) == 2 for label in cases.audit_record(4).labels)
    assert any(call.claimed_model == UNLISTED for call in cases.audit_record(5).calls)
    assert len(cases.audit_record(6).calls) == 20 and set(verdicts(6)) == set(Verdict)


def test_a_near_miss_differs_by_one_small_change() -> None:
    for outputs in REFERENCES.values():
        for text in outputs.values():
            for mutation in MUTATIONS:
                missed = near_miss(text, mutation)
                assert missed != text
                if mutation == "word":
                    pairs = list(zip(missed.split(), text.split()))
                    assert len(missed.split()) == len(text.split()) and sum(a != b for a, b in pairs) == 1
                else:
                    assert missed.rstrip(" .!") == text.rstrip(" .!")
    assert near_miss("Canberra.", "punctuation") == "Canberra"
    assert near_miss("13", "trailing space") == "13 "
    assert near_miss("17 * 23 = 391", "word") == "17 * 23 = 381"


def test_the_generator_refuses_a_near_miss_that_equals_a_reference() -> None:
    a, b = KNOWN[0], KNOWN[1]
    # atlas's "Canberra." without its full stop is borealis's reference, so no rule could call it unknown.
    spec = CaseSpec(calls=(cases.CallSpec(a, "p1", 0.0, "ember-8b", a, "punctuation"),), tests="clash")
    with pytest.raises(ValueError, match="equals a known reference"):
        build_case(0, spec)
    with pytest.raises(ValueError, match="answer alike"):
        build_case(0, CaseSpec(calls=(), tests="undeclared", shared={"p1": (a,)}))
    assert build_case(0, CaseSpec(calls=(cases.CallSpec(b, "p1", 0.0, b),), tests="ok")).labels[0].verdict is Verdict.GENUINE


def test_compare_output_compares_exactly(tmp_path: Path) -> None:
    # Case 1, c02: borealis's "Canberra", which is atlas's "Canberra." without the full stop.
    compared = compare_output(CompareInput(case=1, call_id="c02", models=list(KNOWN)), tmp_path)
    assert compared.prompt_id == "p1"
    assert [(match.model, match.equal) for match in compared.matches] == [
        ("atlas-7b", False),
        ("borealis-13b", True),
        ("cobalt-34b", False),
        ("dynamo-70b", False),
    ]
    # Case 2, c02: cobalt's reference with a trailing space.
    assert not any(match.equal for match in compare_output(CompareInput(2, "c02", list(KNOWN)), tmp_path).matches)
    # Case 3, c01: atlas's reference exactly, at temperature 0.7; the tool still compares, the rules ignore it.
    assert compare_output(CompareInput(3, "c01", ["atlas-7b"]), tmp_path).matches[0].equal is True
    with pytest.raises(ValueError, match="not a probe"):
        compare_output(CompareInput(3, "c03", ["atlas-7b"]), tmp_path)
    with pytest.raises(ValueError, match="not a known model"):
        compare_output(CompareInput(5, "c01", [UNLISTED]), tmp_path)
    with pytest.raises(ValueError, match="no call"):
        compare_output(CompareInput(1, "c99", ["atlas-7b"]), tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        compare_output(CompareInput(1, "c01", []), tmp_path)


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_following_the_instructions_with_the_tools_gives_the_labels(number: int, tmp_path: Path) -> None:
    """What an agent that follows the instructions does: skip unverifiable calls, compare the rest."""
    probes = read_probes(CaseInput(case=number), tmp_path)
    probe_ids = {probe.prompt_id for probe in probes.probes}
    found = []
    for call in read_log(CaseInput(case=number), tmp_path).calls:
        if call.temperature != 0 or call.prompt_id not in probe_ids:
            found.append((call.call_id, Verdict.UNVERIFIABLE, []))
            continue
        compared = compare_output(CompareInput(number, call.call_id, probes.models), tmp_path)
        matched = sorted(match.model for match in compared.matches if match.equal)
        verdict = (
            Verdict.UNKNOWN if not matched else Verdict.GENUINE if call.claimed_model in matched else Verdict.SWAPPED
        )
        found.append((call.call_id, verdict, matched))
    assert found == labels(audit_record(CaseInput(case=number), tmp_path))
