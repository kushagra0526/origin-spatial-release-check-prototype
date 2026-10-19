import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hest_audit import audit  # noqa: E402
from stcheck.audit import BLOCK_T, block_scores, confounding, grouped_split, groups, id_audit, maps  # noqa: E402
from stcheck.synth import S, cohort, public, world  # noqa: E402


def test_id_audit_blank_and_reused():
    meta = [dict(slide="a", site=0, patient="P001", label=1), dict(slide="b", site=1, patient="P001", label=0),
            dict(slide="c", site=1, patient="", label=0), dict(slide="d", site=0, patient="P001", label=1)]
    a = id_audit(meta)
    assert a["blank_patient"] == ["c"] and a["reused_ids"] == ["P001"] and a["multi_slide_patients"] == 1


def test_shifted_section_matches_and_unrelated_does_not():
    c = cohort(world(5), 6, n_patients=12, alias_rate=0)
    Y = c["Y"]
    shifted = np.roll(Y[0].reshape(S, S, -1), (2, -3), (0, 1)).reshape(1, S * S, -1)
    B = block_scores(maps(np.concatenate([Y, shifted]), S))
    blocks = [m["_block"] for m in c["meta"]]
    assert B[0, -1] > 0.9
    unrelated = [B[i, j] for i in range(len(blocks)) for j in range(len(blocks)) if blocks[i] != blocks[j]]
    assert max(unrelated) < BLOCK_T


def test_groups_catch_aliases_without_wrong_merges():
    c = cohort(world(7), 8, n_patients=30, alias_rate=0.4)
    g, links = groups(public(c)["meta"], c["Y"], S)
    meta = c["meta"]
    for i, m in enumerate(meta):
        if m["_alias"]:
            assert any(g[i] == g[j] for j, o in enumerate(meta) if o["_block"] == m["_block"] and not o["_alias"])
        assert all(meta[j]["_patient"] == m["_patient"] for j in range(len(meta)) if g[j] == g[i])
    assert links and all(l["kind"] == "SAME_BLOCK" for l in links)


def test_grouped_split_keeps_groups_whole():
    g = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
    te = grouped_split(g, seed=3)
    for k in np.unique(g):
        assert len(set(te[g == k])) == 1


def test_confounding_flag():
    rng = np.random.default_rng(0)
    meta, X = [], rng.normal(0, 1, (80, 4, 3))
    for i in range(80):
        site = i % 4
        meta.append(dict(slide=str(i), site=site, patient=f"P{i}", label=bool(site < 2 or rng.random() < 0.1)))
    assert confounding(meta, X, np.arange(80))["flag"]
    for i, m in enumerate(meta):
        m["label"] = bool(rng.random() < 0.5)
    assert not confounding(meta, X, np.arange(80))["flag"]


def test_hest_audit_counts_on_tiny_table():
    row = lambda t, p, d="Cancer": dict(species="Homo sapiens", disease_state=d, patient=p, dataset_title=t)
    rows = [row("A", "Patient 1"), row("A", "Patient 1"), row("B", "Patient 1"), row("B", ""), row("C", "x", "Healthy")]
    a = audit(rows, reps=5)
    assert a["cancer_blank_patient"] == 1 and a["patient_strings_reused_across_studies"] == 1
    assert a["cancer_patients_with_2plus_samples"] == 1 and a["cancer_strings_merging_patients_from_different_studies"] == 1
