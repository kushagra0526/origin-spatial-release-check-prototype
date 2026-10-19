"""Release check for a paired H&E + spatial transcriptomics dataset.

Input: per-slide metadata (slide, site, patient, label), spot H&E features X (slides x spots x D) and
spot expression Y (slides x spots x G) on the same spot grid.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter

N_PC = 5
MAX_SHIFT = 4     # each serial section can sit up to 2 spots off, so two can differ by 4
SMOOTH = 1.0      # spots
MUTUAL_K = 4      # a block yields at most a handful of sections: both slides must rank each other top-4
BLOCK_T = 0.72    # map correlation above this = same tissue block (serial sections)


def id_audit(meta):
    """Problems in the patient field itself."""
    blank = [m["slide"] for m in meta if not str(m.get("patient") or "").strip()]
    sites_per_id = defaultdict(set)
    for m in meta:
        if str(m.get("patient") or "").strip():
            sites_per_id[m["patient"]].add(m["site"])
    reused = sorted(p for p, s in sites_per_id.items() if len(s) > 1)
    per_key = Counter((m["site"], m["patient"]) for m in meta if str(m.get("patient") or "").strip())
    return dict(slides=len(meta), blank_patient=blank, reused_ids=reused,
                multi_slide_patients=sum(v > 1 for v in per_key.values()),
                slides_in_multi=sum(v for v in per_key.values() if v > 1))


def maps(Y, side):
    """Per-slide tissue maps: spot expression minus the slide mean, on the top shared PCs."""
    Z = Y - Y.mean(1, keepdims=True)                 # removes site batch and patient offset
    flat = Z.reshape(-1, Z.shape[-1])
    sample = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 20_000), replace=False)]
    _, _, Vt = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    M = (Z @ Vt[:N_PC].T).reshape(len(Y), side, side, N_PC).transpose(0, 3, 1, 2)
    M = gaussian_filter(M, sigma=(0, 0, SMOOTH, SMOOTH), mode="nearest")  # tissue regions span several spots
    M = (M - M.mean((2, 3), keepdims=True)) / (M.std((2, 3), keepdims=True) + 1e-9)
    return M


def block_scores(M, max_shift=MAX_SHIFT):
    """Best Pearson correlation between every pair of slides' tissue maps over small shifts.
    For each shift, both maps are cropped to their overlap and correlated exactly (one matmul)."""
    n, C, s, _ = M.shape
    out = np.full((n, n), -1.0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            a = M[:, :, max(0, dy):s + min(0, dy), max(0, dx):s + min(0, dx)].reshape(n, -1)
            b = M[:, :, max(0, -dy):s + min(0, -dy), max(0, -dx):s + min(0, -dx)].reshape(n, -1)
            a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
            b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
            out = np.maximum(out, a @ b.T / a.shape[1])  # out[i, j]: slide i shifted against slide j
    out = np.maximum(out, out.T)
    return out


def groups(meta, Y, side):
    """Split groups: same (site, patient) key, plus found same-block links (serial sections)."""
    n = len(meta)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    first = {}
    for i, m in enumerate(meta):
        key = (m["site"], m["patient"]) if str(m.get("patient") or "").strip() else ("slide", m["slide"])
        if key in first:
            union(i, first[key])
        first.setdefault(key, i)
    B = block_scores(maps(Y, side))
    np.fill_diagonal(B, -1)
    top = np.argsort(-B, 1)[:, :MUTUAL_K]
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            same_key = find(i) == find(j)
            if B[i, j] >= BLOCK_T and j in top[i] and i in top[j]:
                if not same_key:
                    links.append(dict(a=meta[i]["slide"], b=meta[j]["slide"], kind="SAME_BLOCK", score=round(float(B[i, j]), 2)))
                union(i, j)
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots]), links


def grouped_split(group, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(group)
    test_groups = set(rng.permutation(g)[: max(1, int(round(test_frac * len(g))))].tolist())
    return np.array([x in test_groups for x in group])


def confounding(meta, X, group):
    """Site x label association (one row per group) and how well site can be read off H&E features."""
    seen, rows = set(), []
    for m, g in zip(meta, group):
        if g not in seen:
            seen.add(g)
            rows.append((m["site"], bool(m["label"])))
    sites = sorted({s for s, _ in rows})
    tab = np.array([[sum(1 for s, l in rows if s == si and l == li) for li in (False, True)] for si in sites], float)
    v, p = _cramers_v(tab), _perm_p(rows)
    feats = X.mean(1)
    site = np.array([m["site"] for m in meta])
    hits = 0
    for i in range(len(feats)):  # leave-one-out nearest site centroid
        cents = {s: feats[(site == s) & (np.arange(len(feats)) != i)].mean(0) for s in sites}
        hits += min(cents, key=lambda s: np.sum((feats[i] - cents[s]) ** 2)) == site[i]
    return dict(table=tab.astype(int).tolist(), sites=sites, cramers_v=round(v, 2), p=round(p, 4),
                site_from_he_accuracy=round(hits / len(feats), 2), chance=round(1 / len(sites), 2),
                flag=bool(p < 0.05 and v >= 0.3))


def _cramers_v(tab):
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = float(np.sum((tab - exp) ** 2 / np.where(exp > 0, exp, 1)))
    return (chi2 / (n * (min(tab.shape) - 1))) ** 0.5 if n else 0.0


def _perm_p(rows, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([r[0] for r in rows])
    l = np.array([r[1] for r in rows])
    sites = np.unique(s)
    stat = lambda lab: _cramers_v(np.array([[np.sum((s == si) & (lab == li)) for li in (False, True)] for si in sites], float))
    obs = stat(l)
    return (1 + sum(stat(rng.permutation(l)) >= obs - 1e-12 for _ in range(B))) / (B + 1)
