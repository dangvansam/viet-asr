from multitalker_asr.data.pipeline.config import DiarizeConfig
from multitalker_asr.data.pipeline.stages.vad_diarize import VADDiarizeStage


class TestDiarLineParse:
    def test_compact_format(self):
        st = VADDiarizeStage()
        assert st._parse_diar_line("0.160 1.840 speaker_0") == (0.16, 1.84, "speaker_0")

    def test_full_rttm(self):
        st = VADDiarizeStage()
        line = "SPEAKER file 1 2.00 1.50 <NA> <NA> speaker_1 <NA> <NA>"
        assert st._parse_diar_line(line) == (2.0, 3.5, "speaker_1")

    def test_garbage(self):
        st = VADDiarizeStage()
        assert st._parse_diar_line("") is None
        assert st._parse_diar_line("not a line") is None


class TestOverlapMarking:
    def test_overlap_flagged(self):
        st = VADDiarizeStage()
        # speaker_0 [3.2,4.7] overlaps speaker_1 [3.28,3.44] by 0.16s
        turns = [(0.16, 1.84, "speaker_0"), (3.2, 4.72, "speaker_0"), (3.28, 3.44, "speaker_1")]
        cfg = DiarizeConfig(detect_overlap=True, overlap_min_s=0.1)
        flags = st._mark_overlap(turns, cfg)
        assert flags == [False, True, True]

    def test_no_overlap_same_speaker(self):
        st = VADDiarizeStage()
        turns = [(0.0, 2.0, "speaker_0"), (1.0, 3.0, "speaker_0")]  # same speaker, no flag
        cfg = DiarizeConfig(detect_overlap=True, overlap_min_s=0.1)
        assert st._mark_overlap(turns, cfg) == [False, False]

    def test_detect_disabled(self):
        st = VADDiarizeStage()
        turns = [(0.0, 2.0, "a"), (1.0, 3.0, "b")]
        cfg = DiarizeConfig(detect_overlap=False)
        assert st._mark_overlap(turns, cfg) == [False, False]
