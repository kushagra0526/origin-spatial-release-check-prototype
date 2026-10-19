"""Release check on synthetic cohorts: alias detection, and how much each split inflates a
morphology -> expression score compared with brand-new patients.

python evaluate.py dev       # worlds 0-9, used while building
python evaluate.py heldout   # worlds in heldout/seeds.json, frozen in git before the first run
"""
import json
import sys
from pathlib import Path

import numpy as np

from stcheck.audit import confounding, grouped_split, groups, id_audit
from stcheck.model import knn_r
from stcheck.synth import S, cohort, public, world

HERE = Path(__file__).parent
SPLITS = ("random_spot", "random_slide", "patient_id", "audit_groups")


def split_scores(c, group, fresh, seed):
    """For each split: score on its own test part, and the same trained model on brand-new patients.
    Optimism = own-test score - fresh score (same training data, so the gap is pure leakage)."""
    X, Y, meta = c["X"], c["Y"], c["meta"]
    X = X - X.mean(1, keepdims=True)  # per-slide feature centring, a stand-in for stain normalisation
    fresh = dict(fresh, X=fresh["X"] - fresh["X"].mean(1, keepdims=True))
    n, sp = X.shape[:2]
    rng = np.random.default_rng(seed)
    key = np.array([hash((m["site"], m["patient"])) for m in meta])
    spot = rng.random((n, sp)) < 0.2
    masks = dict(
        random_spot=spot,
        random_slide=np.repeat((rng.random(n) < 0.2)[:, None], sp, 1),
        patient_id=np.repeat(grouped_split(key, seed=seed)[:, None], sp, 1),
        audit_groups=np.repeat(grouped_split(group, seed=seed)[:, None], sp, 1),
    )
    fresh_test = [(fresh["X"][i], fresh["Y"][i]) for i in range(len(fresh["X"]))]
    out = {}
    for name, te in masks.items():
        tr = ~te
        Xtr, Ytr = X[tr], Y[tr]  # boolean (slide, spot) mask -> flat spots
        own = [(X[i][te[i]], Y[i][te[i]]) for i in range(n) if te[i].any()]
        out[name] = knn_r(Xtr, Ytr, own, seed=seed)
        out[name + "_fresh"] = knn_r(Xtr, Ytr, fresh_test, seed=seed)
    return out


def exposure(meta, group, seed, reps=20):
    """Share of test slides that have a section of the same true block / patient in train (no model)."""
    n = len(meta)
    key = np.array([hash((m["site"], m["patient"])) for m in meta])
    blk = np.array([m["_block"] for m in meta])
    pat = np.array([m["_patient"] for m in meta])
    out = {k: dict(block=[], patient=[]) for k in ("random_slide", "patient_id", "audit_groups")}
    for r in range(reps):
        rng = np.random.default_rng(seed * 100 + r)
        masks = dict(random_slide=rng.random(n) < 0.2, patient_id=grouped_split(key, seed=seed * 100 + r),
                     audit_groups=grouped_split(group, seed=seed * 100 + r))
        for k, te in masks.items():
            tb, tp = set(blk[~te]), set(pat[~te])
            out[k]["block"].append(np.mean([b in tb for b in blk[te]]))
            out[k]["patient"].append(np.mean([p in tp for p in pat[te]]))
    return {k: {kk: round(float(np.mean(v)), 3) for kk, v in d.items()} for k, d in out.items()}


def run(world_seeds, confounded_every=2):
    rows = []
    for k, ws in enumerate(world_seeds):
        w = world(ws)
        conf = k % confounded_every == 0
        c = cohort(w, ws + 1, confounded=conf)
        fresh = cohort(w, ws + 2)                       # new patients from the same sites
        pub = public(c)
        group, links = groups(pub["meta"], pub["Y"], S)
        meta = c["meta"]
        alias = [i for i, m in enumerate(meta) if m["_alias"]]
        caught = sum(any(group[i] == group[j] for j, m in enumerate(meta)
                         if m["_block"] == meta[i]["_block"] and not m["_alias"]) for i in alias)
        # slides put in a group with a different true patient
        wrong = sum(any(group[i] == group[j] and meta[j]["_patient"] != meta[i]["_patient"] for j in range(len(meta)))
                    for i in range(len(meta)))
        sc = split_scores(pub, group, fresh, seed=ws)
        exp = exposure(meta, group, seed=ws)
        cf = confounding(pub["meta"], pub["X"], group)
        rows.append(dict(world=ws, slides=len(meta), alias_slides=len(alias), alias_caught=caught,
                         slides_wrongly_merged=wrong, links=len(links), confounded=conf, conf_flag=cf["flag"],
                         cramers_v=cf["cramers_v"], site_from_he=cf["site_from_he_accuracy"],
                         ids=dict((k_, v if isinstance(v, int) else len(v)) for k_, v in id_audit(pub["meta"]).items()),
                         r={k_: round(v, 3) for k_, v in sc.items()}, exposure=exp))
    return rows


def summarize(rows):
    r = {k: round(float(np.mean([x["r"][k] for x in rows])), 3) for k in SPLITS}
    rf = {k: round(float(np.mean([x["r"][k + "_fresh"] for x in rows])), 3) for k in SPLITS}
    infl = {k: round(float(np.mean([x["r"][k] - x["r"][k + "_fresh"] for x in rows])), 3) for k in SPLITS}
    return dict(
        cohorts=len(rows), slides=sum(x["slides"] for x in rows),
        alias_slides=sum(x["alias_slides"] for x in rows), alias_caught=sum(x["alias_caught"] for x in rows),
        slides_wrongly_merged=sum(x["slides_wrongly_merged"] for x in rows),
        mean_r_own_test=r, mean_r_fresh_patients=rf, optimism=infl,
        test_slides_with_same_block_in_train={k: round(float(np.mean([x["exposure"][k]["block"] for x in rows])), 3)
                                               for k in ("random_slide", "patient_id", "audit_groups")},
        test_slides_with_same_patient_in_train={k: round(float(np.mean([x["exposure"][k]["patient"] for x in rows])), 3)
                                                 for k in ("random_slide", "patient_id", "audit_groups")},
        audit_optimism_smallest=sum(min(SPLITS, key=lambda k: abs(x["r"][k] - x["r"][k + "_fresh"])) == "audit_groups" for x in rows),
        confounding=dict(planted=sum(x["confounded"] for x in rows),
                         flagged_when_planted=sum(x["conf_flag"] for x in rows if x["confounded"]),
                         flagged_when_not=sum(x["conf_flag"] for x in rows if not x["confounded"])),
        site_from_he_accuracy=round(float(np.mean([x["site_from_he"] for x in rows])), 2),
    )


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    seeds = list(range(10)) if split == "dev" else json.loads((HERE / "heldout/seeds.json").read_text())["seeds"]
    rows = run(seeds)
    s = summarize(rows)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / f"out/{split}.json").write_text(json.dumps(dict(summary=s, rows=rows), indent=1))
    print(json.dumps(s, indent=1))
