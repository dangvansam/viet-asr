import numpy as np
import pytest

from multitalker_asr.data.pipeline.speaker import (
    assign,
    centroids,
    cluster_embeddings,
    consistency_scores,
    cosine,
    l2_normalize,
)


def _voice(seed, dim=32):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestCosine:
    def test_identical(self):
        v = _voice(1)
        assert cosine(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_zero(self):
        assert cosine(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(0.0)


class TestCluster:
    def test_two_voices_separate(self):
        a, b = _voice(1), _voice(2)
        embs = np.stack([a + 0.02 * _voice(10), a + 0.02 * _voice(11), a, b])
        labels = cluster_embeddings(embs, threshold=0.6)
        # first 3 (voice a) share a label, last (voice b) differs
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]
        assert len(set(labels.tolist())) == 2

    def test_single(self):
        assert cluster_embeddings(_voice(1)[None, :]).tolist() == [0]

    def test_empty(self):
        assert cluster_embeddings(np.zeros((0, 8))).shape == (0,)


class TestConsistency:
    def test_outlier_in_cluster_low_score(self):
        a = _voice(1)
        # force 3 into one cluster but one is an outlier voice
        embs = np.stack([a, a, _voice(99)])
        labels = np.array([0, 0, 0])
        scores = consistency_scores(embs, labels)
        # outlier has the lowest within-cluster agreement
        assert scores[2] < scores[0] and scores[2] < scores[1]
        assert scores[2] < 0.5

    def test_clean_cluster_high_score(self):
        a = _voice(1)
        embs = np.stack([a + 0.02 * _voice(10), a + 0.02 * _voice(11), a])
        labels = np.array([0, 0, 0])
        scores = consistency_scores(embs, labels)
        assert all(s > 0.9 for s in scores)

    def test_singleton_is_one(self):
        assert consistency_scores(_voice(1)[None, :], np.array([0])) == [1.0]


class TestAssign:
    def test_nearest_centroid(self):
        a, b = _voice(1), _voice(2)
        embs = np.stack([a, a, b])
        labels = np.array([0, 0, 1])
        cents = centroids(embs, labels)
        label, sim = assign(a, cents)
        assert label == 0 and sim > 0.9
