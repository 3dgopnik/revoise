import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from core import pipeline


def test_merge_into_phrases_sorts_segments():
    segs = [
        (1.5, 1.6, "C"),
        (0.0, 0.4, "A"),
        (0.4, 1.0, "B"),
    ]
    res = pipeline.merge_into_phrases(segs, max_gap=0.35, min_len=0.8)
    assert res == [
        (0.0, 1.0, "A B"),
        (1.5, 1.6, "C"),
    ]


def test_merge_into_phrases_handles_russian_text():
    segs = [
        (0.0, 0.5, "Привет"),
        (0.5, 1.0, "мир!"),
        (1.5, 2.0, "До"),
        (2.0, 2.5, "свидания"),
    ]
    res = pipeline.merge_into_phrases(segs, max_gap=0.4, min_len=0.1)
    assert res == [
        (0.0, 1.0, "Привет мир!"),
        (1.5, 2.5, "До свидания"),
    ]
