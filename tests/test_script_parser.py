import pytest

from core.script_parser import parse_script, split_chunks


def test_parse_script_valid():
    script = "Speaker 1: Hello world\nSpeaker 2: Hi there"
    lines = parse_script(script)
    assert [line.speaker for line in lines] == [1, 2]
    assert [line.text for line in lines] == ["Hello world", "Hi there"]


def test_parse_script_invalid():
    with pytest.raises(ValueError):
        parse_script("Speaker 5: Too many")


def test_split_chunks():
    # 80 short lines ~80 words -> should split into at least two chunks at max_sec=60
    script = "\n".join(f"Speaker 1: word {i}" for i in range(80))
    lines = parse_script(script)
    chunks = split_chunks(lines, min_sec=30, max_sec=60)
    assert len(chunks) >= 2
