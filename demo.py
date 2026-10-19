"""Release check on one synthetic cohort (dev world 3): 4 partner sites, ~60 patients, planted aliases
and a site-label skew. Runs in ~20 s (no model training; see evaluate.py for that).

python demo.py
"""
from collections import Counter

from evaluate import exposure
from stcheck.audit import confounding, groups, id_audit
from stcheck.synth import S, cohort, public, world

c = cohort(world(3), 4, confounded=True)
pub = public(c)
meta = pub["meta"]
ids = id_audit(meta)
print(f"{ids['slides']} slides. Patient field: {len(ids['blank_patient'])} blank, "
      f"{len(ids['reused_ids'])} IDs reused across sites (P001.. restart per partner -> key on site+patient), "
      f"{ids['multi_slide_patients']} patients with several slides ({ids['slides_in_multi']} slides)\n")

group, links = groups(meta, pub["Y"], S)
truth = {m["slide"]: m for m in c["meta"]}
print(f"Hidden duplicates: {len(links)} slide pairs are serial sections of one block filed under different IDs:")
for l in links[:6]:
    a, b = truth[l["a"]], truth[l["b"]]
    ok = "ok " if a["_block"] == b["_block"] else "XX "
    print(f"  {ok}{l['a']} (site {a['site']}, {a['patient']})  ==  {l['b']} (site {b['site']}, {b['patient']})   map corr {l['score']}")
print(f"  ... {len(links) - 6} more" if len(links) > 6 else "")
print(f"Split groups: {len(set(group))} (from {len(Counter((m['site'], m['patient']) for m in meta))} site+patient keys)\n")

e = exposure(c["meta"], group, seed=3)
print("Test slides with a section of the SAME BLOCK in train (80/20 split, 20 draws):")
for k, label in (("random_slide", "random by slide"), ("patient_id", "by site+patient ID"), ("audit_groups", "by audit groups")):
    print(f"  {label:<20} {e[k]['block']:>6.1%}   same patient: {e[k]['patient']:.1%}")

cf = confounding(meta, pub["X"], group)
print(f"\nSite x responder (one row per group): {dict(zip(cf['sites'], cf['table']))}  "
      f"Cramer's V {cf['cramers_v']}, p {cf['p']} -> {'FLAG: evaluate with a held-out site' if cf['flag'] else 'ok'}")
print(f"Site can be read off H&E features for {cf['site_from_he_accuracy']:.0%} of slides (chance {cf['chance']:.0%}).")
