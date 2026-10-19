"""Synthetic paired H&E-feature + spatial-expression cohorts with planted truth. The audit never imports this.

A *world* fixes the biology and the sites (gene loadings per tissue program, how programs look in H&E
features, each site's stain shift and expression batch). A *cohort* draws patients from that world:
patient -> 1-2 FFPE blocks -> 1-3 serial sections (slides) of 32x32 spots. Serial sections of a block
share the same tissue map, shifted by up to 2 spots. Each patient also has a "phenotype" offset seen in
both modalities, and a tumour clone with its own expression and morphology loadings inside the tumour
program: the part of morphology -> expression that a model can only learn from the same patient.

Planted metadata problems:
* alias: extra serial sections of an existing block filed under a new patient ID (it came via a
  second partner, e.g. a biobank recut)
* reused IDs: patient strings restart at P001 at every site
* site-label skew (in confounded cohorts): some sites send mostly responders
"""
import numpy as np

S, PAD = 32, 4          # spots per side, canvas margin for shifts
G, D, L = 40, 24, 6     # genes, H&E feature dims, tissue programs
SITES = 4
SHARP = 3.0
CLONE = 0.8


def world(seed):
    rng = np.random.default_rng(seed)
    return dict(
        W=rng.normal(0, 1.0, (G, L)),           # program -> log expression
        A=rng.normal(0, 1.0, (D, L)),           # program -> H&E features
        Mz=rng.normal(0, 0.45, (D, 4)),         # patient phenotype -> H&E
        Nz=rng.normal(0, 0.45, (G, 4)),         # patient phenotype -> expression
        stain=rng.normal(0, 0.8, (SITES, D)),   # site stain / scanner shift
        batch=rng.normal(0, 0.3, (SITES, G)),   # site pre-analytic expression shift
        depth=rng.uniform(0.7, 1.3, SITES),
    )


def _tissue(rng, immune_bias):
    n = S + 2 * PAD
    yy, xx = np.mgrid[0:n, 0:n]
    logits = np.zeros((n, n, L))
    for l in range(L):
        for _ in range(3):
            cy, cx, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(2.5, 6)
            logits[:, :, l] += rng.uniform(0.8, 2.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    logits[:, :, 2] += immune_bias  # program 2 = immune infiltrate
    logits *= SHARP                 # distinct regions (tumour nest, stroma, immune aggregate), not a uniform mix
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def cohort(w, seed, n_patients=60, alias_rate=0.15, confounded=False):
    rng = np.random.default_rng(seed)
    slides, X, Y = [], [], []
    patients = []
    for p in range(n_patients):
        site = int(rng.integers(SITES))
        immune = rng.normal(0, 0.8)
        resp = immune + rng.normal(0, 0.5) > 0
        if confounded and site in (0, 1) and not resp and rng.random() < 0.75:
            continue  # sites 0-1 mostly send responders
        patients.append(dict(site=site, immune=immune, resp=bool(resp), z=rng.normal(0, 1, 4)))
    counters = [0] * SITES
    for pi, pt in enumerate(patients):
        counters[pt["site"]] += 1
        pid = f"P{counters[pt['site']]:03d}"  # restarts at every site
        u = w["Nz"] @ pt["z"]
        v = w["Mz"] @ pt["z"]
        # tumour clone of this patient: its own expression and morphology inside the tumour program
        # (e.g. copy-number-driven genes + nuclear features). Learnable from the same patient, useless on others.
        dW = rng.normal(0, CLONE, G)
        dA = rng.normal(0, CLONE, D)
        n_blocks = 1 + int(rng.random() < 0.5)
        alias_block = int(rng.integers(n_blocks)) if rng.random() < alias_rate else -1
        for b in range(n_blocks):
            tissue = _tissue(rng, pt["immune"])
            filings = [(pt["site"], pid, int(rng.integers(1, 4)), False)]
            if b == alias_block:  # the same block re-sectioned via another partner, filed under a new ID
                filings.append((int(rng.integers(SITES)), f"BB-{int(rng.integers(10_000, 99_999))}",
                                int(rng.integers(1, 3)), True))
            for site_meta, pid_meta, n_sec, is_alias in filings:
                for _ in range(n_sec):
                    dy, dx = rng.integers(-2, 3, 2)
                    pi_ = tissue[PAD + dy:PAD + dy + S, PAD + dx:PAD + dx + S].reshape(-1, L)
                    tum = pi_[:, :1]  # program 0 = tumour
                    rate = np.exp(pi_ @ w["W"].T + tum * dW + u + w["batch"][site_meta]) * w["depth"][site_meta]
                    expr = np.log1p(rng.poisson(rate * 3.0))
                    feat = pi_ @ w["A"].T + tum * dA + v + w["stain"][site_meta] + rng.normal(0, 0.5, (S * S, D))
                    slides.append(dict(slide=f"S{len(slides):04d}", site=site_meta, patient=pid_meta, label=pt["resp"],
                                       _patient=pi, _block=f"{pi}-{b}", _alias=is_alias))
                    X.append(feat.astype(np.float32))
                    Y.append(expr.astype(np.float32))
    return dict(meta=slides, X=np.stack(X), Y=np.stack(Y))


def public(c):
    """What a dataset release would contain: no underscore truth fields."""
    return dict(meta=[{k: v for k, v in m.items() if not k.startswith("_")} for m in c["meta"]], X=c["X"], Y=c["Y"])
