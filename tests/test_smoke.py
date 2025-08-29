from core.pipeline import merge_into_phrases


def test_smoke_merge():
    segments = [(0.0, 0.1, "hello"), (0.2, 0.3, "world")]
    assert merge_into_phrases(segments) == [(0.0, 0.3, "hello world")]
