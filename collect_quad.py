#!/usr/bin/env python3
"""Gather every Results/<mol>/<method>/bl_*/energy_summary.json into one CSV.

Run after the quadrature jobs finish (from the dir that contains Results/):
    python collect_quad.py            # -> pes_energies.csv
"""
import csv
import glob
import json
import os

rows = []
for f in glob.glob("Results/*/*/bl_*/energy_summary.json"):
    with open(f) as fh:
        d = json.load(fh)
    parts = f.split(os.sep)              # Results, <mol>, <method>, bl_<bl>, energy_summary.json
    d["method"] = parts[2]
    d["dir"] = os.path.dirname(f)
    rows.append(d)

rows.sort(key=lambda r: (r.get("mol_name", ""), r.get("method", ""),
                         float(r.get("bond_length", 0) or 0)))

cols = ["mol_name", "method", "bond_length", "epoch", "E_total",
        "T", "V_N", "V_H", "E_X", "E_C", "E_CC", "E_NN", "Ne_integral", "dir"]

with open("pes_energies.csv", "w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"wrote pes_energies.csv  ({len(rows)} runs)")
