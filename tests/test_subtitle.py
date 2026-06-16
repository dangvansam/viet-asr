import pytest

from multitalker_asr.utils.subtitle import VTTParser

_VTT = """WEBVTT
Kind: captions
Language: vi

00:00:00.040 --> 00:00:02.629 align:start position:0%
à<00:00:00.520><c> trước</c><00:00:00.760><c> ông</c>

00:00:02.629 --> 00:00:05.000 align:start position:0%
xin chào các bạn

00:00:05.000 --> 00:00:07.500
hôm nay trời đẹp
"""


@pytest.fixture
def vtt_file(tmp_path):
    p = tmp_path / "sub.vi.vtt"
    p.write_text(_VTT, encoding="utf-8")
    return str(p)


class TestVTTParser:
    def test_parse_cues(self, vtt_file):
        cues = VTTParser().parse(vtt_file)
        assert len(cues) == 3
        assert cues[0].start == pytest.approx(0.04)
        assert cues[1].text == "xin chào các bạn"

    def test_inline_tags_stripped(self, vtt_file):
        cues = VTTParser().parse(vtt_file)
        assert "<" not in cues[0].text
        assert "trước" in cues[0].text and "ông" in cues[0].text

    def test_full_text(self, vtt_file):
        text = VTTParser().text(vtt_file)
        assert "xin chào các bạn" in text
        assert "hôm nay trời đẹp" in text

    def test_missing_file(self):
        assert VTTParser().parse("/no/such.vtt") == []
        assert VTTParser().text("/no/such.vtt") == ""
