"""ID audit of a real public dataset's metadata: HEST-1k v1.1.0 (mahmoodlab/HEST, assets/HEST_v1_1_0.csv).
Downloaded at run time, not redistributed. Metadata only: no images or expression are used.

python hest_audit.py                    # downloads the CSV
python hest_audit.py HEST_v1_1_0.csv    # or a copy you fetched with curl
"""
import csv
import io
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/mahmoodlab/HEST/main/assets/HEST_v1_1_0.csv"


def load(path=None):
    if path:
        return list(csv.DictReader(open(path, encoding="utf-8-sig")))
    with urllib.request.urlopen(URL, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8-sig"))))


def audit(rows, reps=200, seed=0):
    human = [r for r in rows if r["species"] == "Homo sapiens"]
    cancer = [r for r in human if r["disease_state"].strip().lower() == "cancer"]
    has = lambda r: r["patient"].strip() != ""
    studies_per_string = defaultdict(set)
    for r in human:
        if has(r):
            studies_per_string[r["patient"].strip()].add(r["dataset_title"])
    reused = {p: len(s) for p, s in studies_per_string.items() if len(s) > 1}
    per_patient = Counter((r["dataset_title"], r["patient"].strip()) for r in cancer if has(r))
    # random 80/20 split by sample: how many test samples have the same (study, patient) in train?
    ids = [r for r in cancer if has(r)]
    keys = np.array([hash((r["dataset_title"], r["patient"].strip())) for r in ids])
    rng = np.random.default_rng(seed)
    leak = []
    for _ in range(reps):
        te = rng.random(len(ids)) < 0.2
        train = set(keys[~te])
        leak.append(np.mean([k in train for k in keys[te]]))
    # grouping by the patient string alone merges different people from different studies
    by_string = Counter(r["patient"].strip() for r in cancer if has(r))
    merged = sum(1 for p in by_string if len({r["dataset_title"] for r in cancer if r["patient"].strip() == p}) > 1)
    return dict(
        human_samples=len(human), human_blank_patient=sum(not has(r) for r in human),
        cancer_samples=len(cancer), cancer_blank_patient=sum(not has(r) for r in cancer),
        cancer_patients_with_ids=len(per_patient),
        cancer_patients_with_2plus_samples=sum(v > 1 for v in per_patient.values()),
        cancer_samples_in_those=sum(v for v in per_patient.values() if v > 1),
        patient_strings_reused_across_studies=len(reused),
        human_samples_with_a_reused_string=sum(1 for r in human if r["patient"].strip() in reused),
        examples_of_reused=sorted(reused, key=lambda p: -reused[p])[:6],
        random_sample_split_test_with_same_patient_in_train=round(float(np.mean(leak)), 3),
        cancer_strings_merging_patients_from_different_studies=merged,
    )


if __name__ == "__main__":
    out = audit(load(sys.argv[1] if len(sys.argv) > 1 else None))
    Path(__file__).with_name("out").mkdir(exist_ok=True)
    Path(__file__).with_name("out").joinpath("hest_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
