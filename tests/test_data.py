from pathlib import Path

from cmr.data import _CAPTION_RE, parse_captions


def test_caption_regex():
    m = _CAPTION_RE.match("100024841.jpg#0\tA child in a pink dress is climbing up a set of stairs .")
    assert m is not None
    assert m.group("image") == "100024841.jpg"
    assert m.group("idx") == "0"
    assert m.group("caption").startswith("A child")


def test_parse_captions_skips_malformed(tmp_path: Path):
    path = tmp_path / "Flickr8k.token.txt"
    path.write_text(
        "a.jpg#0\tfirst caption\n"
        "a.jpg#1\tsecond caption\n"
        "bad line without pattern\n"
        "b.jpg#0\tanother caption\r\n",
        encoding="utf-8",
    )
    captions = parse_captions(path)
    assert captions == {
        "a.jpg": ["first caption", "second caption"],
        "b.jpg": ["another caption"],
    }
