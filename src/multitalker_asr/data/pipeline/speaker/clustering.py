"""
Speaker-embedding clustering + consistency utilities.

Used by the vad_sv diarizer (assign speaker labels from embeddings) and the
speaker_verify stage (re-cluster to verify/correct diarization labels). Cosine
similarity throughout; same-speaker threshold defaults to 0.725 (the value the
speaker-recognition service uses).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.725) -> np.ndarray:
    """Agglomerative clustering by cosine; clusters merge while cosine >= threshold.

    Returns an integer label per row. N<=1 → all zeros.
    """
    embs = np.asarray(embeddings, dtype=np.float32)
    n = embs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int)
    if n == 1:
        return np.zeros((1,), dtype=int)

    normed = l2_normalize(embs)
    try:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
        return model.fit_predict(normed).astype(int)
    except Exception:
        return _greedy_cluster(normed, threshold)


def _greedy_cluster(normed: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomerative fallback (no sklearn). O(N^2), small N only."""
    n = normed.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]

    def cluster_sim(a: List[int], b: List[int]) -> float:
        sims = [float(np.dot(normed[i], normed[j])) for i in a for j in b]
        return sum(sims) / len(sims)

    while len(clusters) > 1:
        best = (-1.0, 0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = cluster_sim(clusters[i], clusters[j])
                if s > best[0]:
                    best = (s, i, j)
        if best[0] < threshold:
            break
        _, i, j = best
        clusters[i].extend(clusters[j])
        del clusters[j]

    labels = np.zeros(n, dtype=int)
    for label, members in enumerate(clusters):
        for idx in members:
            labels[idx] = label
    return labels


def centroids(embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    normed = l2_normalize(embeddings)
    out: Dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        members = normed[labels == label]
        out[int(label)] = l2_normalize(members.mean(axis=0, keepdims=True))[0]
    return out


def assign(emb: np.ndarray, cluster_centroids: Dict[int, np.ndarray]) -> Tuple[Optional[int], float]:
    """Nearest cluster by cosine. Returns (label, cosine) or (None, 0.0)."""
    best_label, best_sim = None, -1.0
    for label, centroid in cluster_centroids.items():
        sim = cosine(emb, centroid)
        if sim > best_sim:
            best_label, best_sim = label, sim
    if best_label is None:
        return None, 0.0
    return best_label, best_sim


def cluster_margin(emb: np.ndarray, label: int, cluster_centroids: Dict[int, np.ndarray]) -> float:
    """cos to own centroid minus the best cos to any OTHER centroid.

    High margin = confidently in its cluster; near 0 = sits between clusters
    (ambiguous). With a single cluster, returns the cos to that centroid.
    """
    own = cosine(emb, cluster_centroids[label])
    others = [cosine(emb, c) for k, c in cluster_centroids.items() if k != label]
    if not others:
        return round(own, 4)
    return round(own - max(others), 4)


def estimate_threshold(
    embeddings: np.ndarray,
    floor: float = 0.30,
    ceil: float = 0.65,
    default: float = 0.45,
) -> float:
    """Per-file cosine threshold from the pairwise-similarity distribution.

    Finds the largest gap between consecutive sorted pairwise cosines that fall
    in [floor, ceil] and puts the threshold at the gap midpoint (the natural
    valley between same- and different-speaker pairs). Falls back to `default`.
    """
    embs = l2_normalize(embeddings)
    n = embs.shape[0]
    if n < 3:
        return default
    sims = embs @ embs.T
    pair = np.sort(sims[np.triu_indices(n, 1)])
    band = pair[(pair >= floor) & (pair <= ceil)]
    if band.size < 2:
        return default
    gaps = np.diff(band)
    i = int(np.argmax(gaps))
    return round(float((band[i] + band[i + 1]) / 2.0), 4)


def consistency_scores(embeddings: np.ndarray, labels: np.ndarray) -> List[float]:
    """Per-row leave-one-out mean cosine to other members of its cluster.

    Cluster of size 1 → 1.0 (cannot verify). Robust to a single outlier.
    """
    normed = l2_normalize(embeddings)
    n = normed.shape[0]
    scores: List[float] = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not same:
            scores.append(1.0)
            continue
        sims = [float(np.dot(normed[i], normed[j])) for j in same]
        scores.append(sum(sims) / len(sims))
    return scores
