"""Utilities for parsing speaker-labelled scripts.

Each non-empty line must start with ``Speaker N:`` where ``N`` is in ``1..4``.
The parser returns a list of ``Line`` objects and supports splitting scripts
into chunks roughly matching a target duration.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

SPEAKER_RE = re.compile(r"^Speaker ([1-4]):\s*(.*)$")


@dataclass
class Line:
    """A single parsed script line."""

    speaker: int
    text: str


def parse_script(script: str) -> list[Line]:
    """Parse ``script`` into ``Line`` objects.

    Parameters
    ----------
    script:
        Raw multiline script where each line is ``Speaker N: text``.

    Raises
    ------
    ValueError
        If a line does not match the expected pattern or speaker number.
    """

    lines: list[Line] = []
    for lineno, raw in enumerate(script.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        match = SPEAKER_RE.match(stripped)
        if not match:
            raise ValueError(f"Invalid line {lineno!r}: {raw!r}")
        speaker = int(match.group(1))
        text = match.group(2).strip()
        if not text:
            raise ValueError(f"Empty text for speaker {speaker} at line {lineno}")
        lines.append(Line(speaker, text))
    return lines


def split_chunks(lines: Sequence[Line], *, min_sec: float = 30.0, max_sec: float = 120.0) -> list[list[Line]]:
    """Split ``lines`` into duration-based chunks.

    Duration is estimated assuming 150 words per minute (~2.5 words/s).
    """

    if min_sec <= 0 or max_sec <= 0:
        raise ValueError("Durations must be positive")
    if min_sec >= max_sec:
        raise ValueError("min_sec must be < max_sec")

    words_per_sec = 150 / 60.0  # 150 wpm
    min_words = int(min_sec * words_per_sec)
    max_words = int(max_sec * words_per_sec)

    chunks: list[list[Line]] = []
    cur: list[Line] = []
    cur_words = 0

    for line in lines:
        words = len(line.text.split())
        if cur and cur_words + words > max_words and cur_words >= min_words:
            chunks.append(cur)
            cur = []
            cur_words = 0
        cur.append(line)
        cur_words += words
    if cur:
        chunks.append(cur)
    return chunks
