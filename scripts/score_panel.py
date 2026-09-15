"""Score the external panel and write one cell file per corpus.

A cell is one system, one elicitation rung and one quantity, pooled over the sampling seeds by
clip, so n counts distinct clips carrying a parsed claim.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acousticlaim.parser import ClaimParser
from acousticlaim.scoring import (BAR, MIN_CLAIMS, MIN_DISTINCT, clip_id, generated_text,
                                  load_reference, parse_claims, score_cell, scored_quantities)

CORPORA = {"librimix": "Libri2Mix", "ami": "AMI"}
REFERENCE_FILES = {"librimix": "librimix_test.csv", "ami": "ami_test.csv"}


def read_generations(path):
    """Yield (system, rung, clip, text) from one elicitation file."""
    with open(path) as handle:
        dump = json.load(handle)
    for system, rungs in (dump.get("results") or {}).items():
        for rung, cell in rungs.items():
            if not isinstance(cell, dict):
                continue
            for record in cell.get("generations") or []:
                yield system, rung, clip_id(record), generated_text(record)


def pool_claims(files, reference, quantities, parser):
    """Pool the parsed claims of every seed into {(system, rung): {clip: {quantity: value}}}."""
    pooled = {}
    clips = set()
    systems = set()
    for path in files:
        for system, rung, clip, text in read_generations(path):
            systems.add(system)
            if clip not in reference:
                continue
            clips.add(clip)
            claims = parse_claims(parser, text, quantities, external=True)
            if not claims:
                continue
            stated = pooled.setdefault((system, rung), {}).setdefault(clip, {})
            for quantity, value in claims.items():
                stated.setdefault(quantity, value)
    return pooled, clips, sorted(systems)


def score_corpus(panel_dir, reference_path, corpus, parser):
    files = sorted(glob.glob(os.path.join(panel_dir, f"*_{corpus}_seed*.json")))
    if not files:
        raise SystemExit(f"no {corpus} panel files in {panel_dir}")
    reference = load_reference(reference_path, corpus)
    quantities = scored_quantities(corpus)
    pooled, clips, systems = pool_claims(files, reference, quantities, parser)
    cells = []
    for (system, rung), per_clip in sorted(pooled.items()):
        for quantity in quantities:
            stated, measured = [], []
            for clip, claims in per_clip.items():
                if quantity in claims and quantity in reference[clip]:
                    stated.append(claims[quantity])
                    measured.append(reference[clip][quantity])
            cell = score_cell(stated, measured, n_clips=len(clips))
            if cell["n"] < MIN_CLAIMS:
                continue
            cells.append({"system": system, "rung": rung, "quantity": quantity, **cell})
    return {"corpus": corpus, "files": [os.path.basename(f) for f in files],
            "clips": len(clips), "systems": systems, "quantities": list(quantities),
            "thresholds": {"min_claims": MIN_CLAIMS, "min_distinct": MIN_DISTINCT, "bar": BAR},
            "cells": cells}


def counts(cells):
    """Cells scored, cells ranked, cells under the distinct-value minimum, cells over the bar."""
    ranked = [c for c in cells if c["status"] == "ranked"]
    return (f"{len(cells)} cells at {MIN_CLAIMS} or more parsed claims, {len(ranked)} ranked, "
            f"{sum(1 for c in cells if c['status'] == 'constant')} under {MIN_DISTINCT} distinct "
            f"values, {sum(1 for c in ranked if c['spearman'] > BAR)} over {BAR}")


def summarise(result):
    cells = result["cells"]
    above = [c for c in cells if c["status"] == "ranked" and c["spearman"] > BAR]
    name = CORPORA[result["corpus"]]
    print(f"\n{name}: {result['clips']} panel clips, {len(result['files'])} files")
    print("  " + counts(cells))
    for system in result["systems"]:
        own = [c for c in cells if c["system"] == system]
        hits = sum(1 for c in own if c["status"] == "ranked" and c["spearman"] > BAR)
        print(f"  {system:<20} {hits}/{len(own)}")
    for cell in sorted(above, key=lambda c: -c["spearman"]):
        print(f"  over the bar  {cell['system']} / {cell['rung']} / {cell['quantity']} "
              f"{cell['spearman']:+.3f} (n {cell['n']}, {cell['distinct']} distinct, "
              f"nMAE {cell['nmae']:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", default="outputs/panel",
                    help="folder holding the elicitation files, one per system, corpus and seed")
    ap.add_argument("--references", default="data/references",
                    help="folder holding librimix_test.csv and ami_test.csv")
    ap.add_argument("--results", default="results", help="folder to write the cell files into")
    ap.add_argument("--corpus", choices=list(CORPORA), action="append",
                    help="score this corpus only, repeatable")
    args = ap.parse_args()

    parser = ClaimParser()
    os.makedirs(args.results, exist_ok=True)
    scored = []
    for corpus in args.corpus or list(CORPORA):
        reference_path = os.path.join(args.references, REFERENCE_FILES[corpus])
        result = score_corpus(args.panel, reference_path, corpus, parser)
        out = os.path.join(args.results, f"panel_{corpus}.json")
        with open(out, "w") as handle:
            json.dump(result, handle, indent=1)
        summarise(result)
        print(f"  wrote {out}")
        scored.append(result)
    if len(scored) > 1:
        print("\nBoth corpora: " + counts([c for r in scored for c in r["cells"]]))


if __name__ == "__main__":
    main()
