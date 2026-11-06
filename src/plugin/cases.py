"""The data generator: known models' reference outputs, and API logs built from the model that really answered.

Each call is built from its source, the model that really answered it, and labelled from what the generator
knows about it: whether its temperature and prompt let a reference apply, which known models share the
source's reference, and whether the source is a known model at all. A model outside the known set answers
with a near miss of a known reference, never with one exactly, so the rules can tell it apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from contract import CASES, Call, Label, Probe, RecordOutput, Reference, Verdict

KNOWN = ("atlas-7b", "borealis-13b", "cobalt-34b", "dynamo-70b")
# Models outside the known set: they answer calls but have no references.
OUTSIDE = ("ember-8b", "fjord-22b")
# A model name an API may claim that is not a known model.
UNLISTED = "helios-405b"

PROBES = {
    "p1": "Name the capital of Australia in one word.",
    "p2": "Give the next prime number after 11.",
    "p3": "Translate 'good morning' into French.",
    "p4": "What is 17 * 23?",
    "p5": "Complete the phrase: the quick brown fox",
}
# Each known model's output for each probe at temperature 0. Some differ only in punctuation.
REFERENCES = {
    "p1": {"atlas-7b": "Canberra.", "borealis-13b": "Canberra", "cobalt-34b": "The capital is Canberra.",
           "dynamo-70b": "Canberra is the capital."},
    "p2": {"atlas-7b": "13", "borealis-13b": "13.", "cobalt-34b": "The next prime is 13.",
           "dynamo-70b": "After 11 comes 13."},
    "p3": {"atlas-7b": "Bonjour.", "borealis-13b": "Bonjour!", "cobalt-34b": "Bonjour",
           "dynamo-70b": "In French: Bonjour."},
    "p4": {"atlas-7b": "391", "borealis-13b": "17 * 23 = 391", "cobalt-34b": "391.",
           "dynamo-70b": "The product is 391."},
    "p5": {"atlas-7b": "jumps over the lazy dog.", "borealis-13b": "The quick brown fox jumps over the lazy dog.",
           "cobalt-34b": "jumps over the lazy dog", "dynamo-70b": "...jumps over the lazy dog."},
}
# Prompts that are not probes, and what any model answers to them.
UNPROBED = {
    "u1": "Rain fell softly on the tin roof all night.",
    "u2": "Try Middlemarch by George Eliot.",
}
# How a model outside the known set misses a reference: one word changed, the final punctuation, a space.
WORD_SWAPS = (("Canberra", "Sydney"), ("13", "17"), ("Bonjour", "Salut"), ("391", "381"), ("dog", "cat"))
MUTATIONS = ("word", "punctuation", "trailing space")


def near_miss(text: str, mutation: str) -> str:
    """``text`` changed in one small way, so it is no longer equal."""
    if mutation == "word":
        for old, new in WORD_SWAPS:
            if old in text:
                return text.replace(old, new, 1)
        raise ValueError(f"no word of {text!r} has a swap")
    if mutation == "punctuation":
        return text[:-1] if text[-1] in ".!" else text + "."
    if mutation == "trailing space":
        return text + " "
    raise ValueError(f"unknown mutation {mutation!r}")


@dataclass(frozen=True)
class CallSpec:
    claimed: str  # the model the API names
    prompt_id: str  # a probe, or an unprobed prompt
    temperature: float
    source: str  # the model that really answered
    # For a source outside the known set, the known model whose reference it misses and how. For a known
    # source at a temperature above 0, how its sampled output differs from its reference, or None.
    base: str | None = None
    mutation: str | None = None


@dataclass(frozen=True)
class CaseSpec:
    calls: tuple[CallSpec, ...]
    tests: str  # what the case is there to test
    # Probes on which known models answer alike: each later model takes the first one's reference.
    shared: dict[str, tuple[str, ...]] = field(default_factory=dict)


def references(shared: dict[str, tuple[str, ...]]) -> dict[str, dict[str, str]]:
    table = {prompt_id: dict(outputs) for prompt_id, outputs in REFERENCES.items()}
    for prompt_id, models in shared.items():
        for model in models[1:]:
            table[prompt_id][model] = table[prompt_id][models[0]]
    return table


def output_of(spec: CallSpec, table: dict[str, dict[str, str]]) -> str:
    if spec.prompt_id in UNPROBED:
        return UNPROBED[spec.prompt_id]
    if spec.source in KNOWN:
        reference = table[spec.prompt_id][spec.source]
        return reference if spec.mutation is None else near_miss(reference, spec.mutation)
    if spec.base is None or spec.mutation is None:
        raise ValueError(f"{spec.source} is outside the known set, so its call needs a base and a mutation")
    return near_miss(table[spec.prompt_id][spec.base], spec.mutation)


def label_of(call_id: str, spec: CallSpec, shared: dict[str, tuple[str, ...]], output: str, table) -> Label:
    """The call's label from how it was built, not from comparing its output."""
    if spec.temperature != 0 or spec.prompt_id not in PROBES:
        return Label(call_id=call_id, source_model=spec.source, verdict=Verdict.UNVERIFIABLE, matched_models=[])
    if spec.source not in KNOWN:
        if output in table[spec.prompt_id].values():
            raise ValueError(f"{call_id}: {spec.source}'s near miss equals a known reference, so no rule can tell")
        return Label(call_id=call_id, source_model=spec.source, verdict=Verdict.UNKNOWN, matched_models=[])
    if spec.mutation is not None:
        raise ValueError(f"{call_id}: a known model answers a probe at temperature 0 with its reference exactly")
    sharing = shared.get(spec.prompt_id, ())
    group = sharing if spec.source in sharing else (spec.source,)
    verdict = Verdict.GENUINE if spec.claimed in group else Verdict.SWAPPED
    return Label(call_id=call_id, source_model=spec.source, verdict=verdict, matched_models=sorted(group))


def build_case(number: int, spec: CaseSpec) -> RecordOutput:
    table = references(spec.shared)
    for prompt_id, outputs in table.items():
        alike = {model for model in KNOWN for other in KNOWN if other != model and outputs[other] == outputs[model]}
        if alike != set(spec.shared.get(prompt_id, ())):
            raise ValueError(f"on {prompt_id} the models {sorted(alike)} answer alike, not the declared ones")
    calls, labels = [], []
    for index, call in enumerate(spec.calls, start=1):
        call_id = f"c{index:02d}"
        output = output_of(call, table)
        calls.append(
            Call(call_id=call_id, claimed_model=call.claimed, prompt_id=call.prompt_id,
                 temperature=call.temperature, output=output)
        )
        labels.append(label_of(call_id, call, spec.shared, output, table))
    return RecordOutput(
        case=number,
        models=sorted(KNOWN),
        probes=[Probe(prompt_id=prompt_id, prompt=prompt) for prompt_id, prompt in PROBES.items()],
        references=[
            Reference(model=model, prompt_id=prompt_id, output=table[prompt_id][model])
            for prompt_id in PROBES
            for model in sorted(KNOWN)
        ],
        calls=calls,
        labels=labels,
    )


def random_calls(rng: random.Random, count: int, shared: dict[str, tuple[str, ...]]) -> tuple[CallSpec, ...]:
    """``count`` calls of every kind, drawn from ``rng``."""
    table = references(shared)
    calls = []
    for _ in range(count):
        claimed = rng.choice(KNOWN) if rng.random() < 0.85 else UNLISTED
        prompt_id = rng.choice(sorted(PROBES)) if rng.random() < 0.85 else rng.choice(sorted(UNPROBED))
        temperature = 0.0 if rng.random() < 0.8 else rng.choice((0.2, 0.7, 1.0))
        roll = rng.random()
        if roll < 0.5 and claimed in KNOWN:
            calls.append(CallSpec(claimed, prompt_id, temperature, claimed))
        elif roll < 0.8:
            mutation = rng.choice((None, "punctuation")) if temperature else None
            calls.append(CallSpec(claimed, prompt_id, temperature, rng.choice(KNOWN), mutation=mutation))
        else:
            # A near miss that equals no known reference: the only kind a model outside the set gives.
            misses = [
                (base, mutation)
                for base in KNOWN
                for mutation in MUTATIONS
                if prompt_id in UNPROBED or near_miss(table[prompt_id][base], mutation) not in table[prompt_id].values()
            ]
            base, mutation = rng.choice(misses)
            calls.append(CallSpec(claimed, prompt_id, temperature, rng.choice(OUTSIDE), base, mutation))
    return tuple(calls)


A, B, C, D = KNOWN
SPECS = {
    1: CaseSpec(
        calls=(
            CallSpec(A, "p1", 0.0, A),
            CallSpec(B, "p1", 0.0, B),
            CallSpec(C, "p3", 0.0, C),
            CallSpec(A, "p2", 0.0, B),
            CallSpec(D, "p4", 0.0, D),
            CallSpec(B, "p5", 0.0, B),
        ),
        tests="mostly genuine calls and one swap to a model whose reference differs only by a full stop",
    ),
    2: CaseSpec(
        calls=(
            CallSpec(A, "p1", 0.0, "ember-8b", A, "word"),
            CallSpec(C, "p2", 0.0, "ember-8b", C, "trailing space"),
            CallSpec(D, "p4", 0.0, D),
            CallSpec(B, "p5", 0.0, "fjord-22b", B, "punctuation"),
            CallSpec(A, "p3", 0.0, A),
            CallSpec(B, "p4", 0.0, "fjord-22b", B, "word"),
        ),
        tests="near misses from models outside the known set, which must be unknown, never the closest model",
    ),
    3: CaseSpec(
        calls=(
            CallSpec(A, "p1", 0.7, A),
            CallSpec(B, "p2", 0.0, B),
            CallSpec(C, "u1", 0.0, C),
            CallSpec(D, "p4", 0.2, D, mutation="punctuation"),
            CallSpec(A, "p5", 0.0, C),
            CallSpec(B, "p3", 1.0, A),
        ),
        tests="calls at a temperature above 0 or on a prompt that is not a probe, even when the output equals a"
        " reference",
    ),
    4: CaseSpec(
        calls=(
            CallSpec(A, "p3", 0.0, A),
            CallSpec(C, "p3", 0.0, A),
            CallSpec(B, "p3", 0.0, C),
            CallSpec(D, "p3", 0.0, D),
            CallSpec(C, "p1", 0.0, C),
            CallSpec(A, "p2", 0.0, D),
        ),
        shared={"p3": (A, C)},
        tests="two known models with the same reference on one probe, so a call matches both",
    ),
    5: CaseSpec(
        calls=(
            CallSpec(UNLISTED, "p1", 0.0, A),
            CallSpec(UNLISTED, "p2", 0.0, "ember-8b", D, "word"),
            CallSpec(A, "p4", 0.0, A),
            CallSpec(UNLISTED, "p5", 0.0, D),
            CallSpec(B, "p1", 0.0, B),
            CallSpec(UNLISTED, "u2", 0.0, UNLISTED),
        ),
        tests="a claimed model that is not a known model",
    ),
    6: CaseSpec(
        calls=random_calls(random.Random(59), 20, {"p2": (B, D)}),
        shared={"p2": (B, D)},
        tests="a mixed log of 20 calls with every verdict and every trap above",
    ),
}


def audit_record(number: int) -> RecordOutput:
    """Built-in case ``number``."""
    if not 1 <= number <= CASES:
        raise ValueError(f"case must be 1 to {CASES}, not {number}")
    return build_case(number, SPECS[number])
