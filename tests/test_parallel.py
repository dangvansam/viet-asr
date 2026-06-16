import pytest

from multitalker_asr.data.pipeline.parallel import parallel_call, parallel_map


class TestParallelMap:
    def test_preserves_input_order(self):
        assert parallel_map(lambda x: x * x, [1, 2, 3, 4], workers=4) == [1, 4, 9, 16]

    def test_empty_returns_empty(self):
        assert parallel_map(lambda x: x, [], workers=8) == []

    def test_single_worker_runs_inline(self):
        assert parallel_map(lambda x: x + 1, [10, 20], workers=1) == [11, 21]

    def test_more_workers_than_items_is_safe(self):
        assert parallel_map(str, [1, 2], workers=16) == ["1", "2"]

    def test_exception_propagates(self):
        def boom(x):
            if x == 2:
                raise ValueError("boom")
            return x
        with pytest.raises(ValueError):
            parallel_map(boom, [1, 2, 3], workers=3)

    def test_parallel_call_runs_thunks_in_order(self):
        assert parallel_call([lambda: 1, lambda: 2, lambda: 3], workers=3) == [1, 2, 3]
