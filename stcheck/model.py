"""A common morphology -> expression baseline: k-nearest neighbours on H&E features (as people do with
foundation-model embeddings). Scored like HEST-Benchmark: Pearson r per gene within each test slide,
averaged over genes and slides, so the score does not depend on how many patients the test set mixes."""
import numpy as np


def knn_r(Xtr, Ytr, test, k=10, max_slides=15, max_spots=300, seed=0):
    """test: list of (X_slide_spots, Y_slide_spots)."""
    rng = np.random.default_rng(seed)
    if len(test) > max_slides:
        test = [test[i] for i in rng.choice(len(test), max_slides, replace=False)]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    a2 = (A * A).sum(1)
    rs = []
    for Xs, Ys in test:
        if len(Xs) > max_spots:
            idx = rng.choice(len(Xs), max_spots, replace=False)
            Xs, Ys = Xs[idx], Ys[idx]
        q = (Xs - mu) / sd
        nn = np.argpartition(a2[None] - 2 * q @ A.T, k, axis=1)[:, :k]
        pred = Ytr[nn].mean(1)
        yc, pc = Ys - Ys.mean(0), pred - pred.mean(0)
        den = np.sqrt((yc * yc).sum(0) * (pc * pc).sum(0))
        ok = den > 0
        rs.append(float(np.mean((yc * pc).sum(0)[ok] / den[ok])))
    return float(np.mean(rs))
