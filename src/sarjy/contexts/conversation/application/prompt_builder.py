from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from sarjy.contexts.conversation.application.ports import Fact
from sarjy.shared.invisible import INVISIBLE_RE

_DEFAULT_STATIC = Path(__file__).parent.parent / "infrastructure" / "prompts" / "system_static.md"
_DELIMS = re.compile(r"</?(facts|user|workflow|results|system)>", re.I)
# Invisible characters an injection can hide inside a delimiter (a zero-width space
# between "us" and "er" keeps <user> off the regex) or use to smuggle instructions
# past a human reviewer. The set is shared with the guard's normaliser and the
# memory sanitiser (`sarjy.shared.invisible`) so all three strip exactly the same
# characters — three private copies had already drifted apart once.
_INVISIBLE = INVISIBLE_RE


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    system: str
    hash: str


def sanitise_value(v: str, limit: int = 200) -> str:
    # NFKC first so compatibility forms (the fullwidth less-than/greater-than signs
    # U+FF1C and U+FF1E, etc.) fold onto the ASCII delimiters, then strip invisibles
    # so they cannot break a delimiter apart, and only then remove the delimiters.
    v = unicodedata.normalize("NFKC", v)
    v = _INVISIBLE.sub("", v)
    # To a fixed point: one pass over "<</user>/user>" removes the inner "</user>" and
    # splices the surviving halves into a fresh one, so a single sub would hand the
    # model back the very delimiter it was supposed to strip.
    while True:
        stripped = _DELIMS.sub("", v)
        if stripped == v:
            break
        v = stripped
    v = re.sub(r"[\r\n\t]+", " ", v)
    return v.strip()[:limit]


class PromptBuilder:
    def __init__(self, static_path: Path = _DEFAULT_STATIC) -> None:
        self._static = static_path.read_text(encoding="utf-8").strip()

    def build(
        self,
        facts: list[Fact],
        workflow_block: str | None,
        summary: str | None,
        results_block: str | None = None,
    ) -> BuiltPrompt:
        """`results_block` is the finished Big Five run's scores (P-9/P-11).

        It is mutually exclusive with `workflow_block` in practice — the loader
        only carries last time's results when no run is open — but both are
        accepted so neither caller has to know that. It gets its own delimiter
        rather than being folded into `<workflow>`: one says what is happening
        now, the other what already happened, and a model told "item 7 of 20"
        and "Openness 2.8" under the same tag has been invited to conflate them.
        """
        parts = [self._static, "<facts>"]
        parts += [f"{sanitise_value(f.key, 60)}: {sanitise_value(f.value)}" for f in facts] or [
            "(none stored)"
        ]
        parts.append("</facts>")
        if workflow_block:
            parts += ["<workflow>", sanitise_value(workflow_block, 600), "</workflow>"]
        if results_block:
            parts += ["<results>", sanitise_value(results_block, 600), "</results>"]
        if summary:
            parts += ["Earlier in this conversation:", sanitise_value(summary, 800)]
        system = "\n".join(parts)
        h = hashlib.sha256(system.encode()).hexdigest()[:12]
        return BuiltPrompt(system=system, hash=h)

    @property
    def static_text(self) -> str:
        return self._static

    @property
    def confidential_text(self) -> str:
        """The confidential portion of the static prompt: everything after
        the persona line and the capability blurb — the POLICY (hard
        rules), GROUNDING, INSTRUCTION HIERARCHY, and TOOL GUIDANCE blocks.

        This is what the output-guard's leak detector should be compared
        against (see `sarjy.contexts.guardrails.domain.leak_detector`) —
        the persona line and capability blurb are things Sarjy is allowed,
        even expected, to restate in conversation, so including them would
        make ordinary replies look like leaks.
        """
        paragraphs = self._static.split("\n\n")
        return "\n\n".join(paragraphs[2:])
