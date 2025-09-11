import pytest

from core.script_parser import parse_script, split_chunks


@pytest.mark.parametrize("speaker_count", [1, 2, 3, 4])
def test_parse_script_handles_up_to_four_speakers(speaker_count):
    script = "\n".join(f"Speaker {i}: line {i}" for i in range(1, speaker_count + 1))
    lines = parse_script(script)
    assert [line.speaker for line in lines] == list(range(1, speaker_count + 1))


@pytest.mark.parametrize(
    "bad_line",
    [
        "Speaker 0: nope",
        "Speaker 5: nope",
        "Speaker one: nope",
        "Speaker 1 nope",
        "Speaker 1:",
    ],
)
def test_parse_script_rejects_invalid_formats(bad_line):
    with pytest.raises(ValueError):
        parse_script(bad_line)


def test_split_chunks():
    # 80 short lines ~80 words -> should split into at least two chunks at max_sec=60
    script = "\n".join(f"Speaker 1: word {i}" for i in range(80))
    lines = parse_script(script)
    chunks = split_chunks(lines, min_sec=30, max_sec=60)
    assert len(chunks) >= 2
