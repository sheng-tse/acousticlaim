"""Score the decoder's emitted text against the instrument, one file per training seed.

A quantity the decoder did not state lowers coverage and contributes no error. The parser is
applied to the generated text only; the reference always comes from the instrument CSV.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acousticlaim.parser import ClaimParser
from acousticlaim.scoring import (MIN_CLAIMS, MIN_DISTINCT, SPLITS, clip_id, generated_text,
                                  in_split, load_reference, mean_sd, parse_claims, score_cell,
                                  scored_quantities)


def score_file(path, reference, quantities, split, parser):
    """Score one seed's generations, one cell per quantity."""
    with open(path) as handle:
        records = json.load(handle)
    stated = {quantity: [] for quantity in quantities}
    measured = {quantity: [] for quantity in quantities}
    clips = 0
    for record in records:
        clip = clip_id(record)
        if clip not in reference or not in_split(clip, split):
            continue
        clips += 1
        claims = parse_claims(parser, generated_text(record), quantities)
        for quantity, value in claims.items():
            if quantity in reference[clip]:
                stated[quantity].append(value)
                measured[quantity].append(reference[clip][quantity])
    if not clips:
        raise SystemExit(f"{path} has no clip of the {split} split in the reference")
    return clips, {quantity: score_cell(stated[quantity], measured[quantity], n_clips=clips)
                   for quantity in quantities}


def aggregate(cells):
    """Mean and s.d. over the seeds whose cell is ranked, with coverage over every seed."""
    ranked = [cell for cell in cells if cell["status"] == "ranked"]
    return {"coverage": mean_sd([cell["coverage"] for cell in cells]),
            "spearman": mean_sd([cell["spearman"] for cell in ranked]),
            "nmae": mean_sd([cell["nmae"] for cell in ranked]),
            "n_per_seed": [cell["n"] for cell in cells],
            "distinct_per_seed": [cell["distinct"] for cell in cells],
            "seeds": len(cells),
            "seeds_ranked": len(ranked),
            "seeds_below_min_claims": sum(1 for cell in cells if cell["n"] < MIN_CLAIMS)}


def fmt(value, form):
    """A number in the given form, or a dash where the statistic does not exist."""
    return "--" if value is None else form % value


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs/decoder/librimix_eval600_seed*.json"],
                    help="generations files, one per training seed")
    ap.add_argument("--reference", default="data/references/librimix_test.csv")
    ap.add_argument("--corpus", default="librimix", choices=["librimix", "ami"],
                    help="which corpus the reference CSV is, for the column names")
    ap.add_argument("--split", default="pooled", choices=list(SPLITS),
                    help="mixtures, their clean twins, or both")
    ap.add_argument("--out",
                    help="file to write, results/decoder_<corpus>_<split>.json by default")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    files = sorted({path for pattern in args.outputs for path in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"no generations files matching {' '.join(args.outputs)}")
    reference = load_reference(args.reference, args.corpus)
    quantities = scored_quantities(args.corpus)
    parser = ClaimParser()

    per_seed, clip_counts = {quantity: [] for quantity in quantities}, []
    for path in files:
        clips, cells = score_file(path, reference, quantities, args.split, parser)
        clip_counts.append(clips)
        for quantity in quantities:
            per_seed[quantity].append(cells[quantity])
    summary = {quantity: aggregate(per_seed[quantity]) for quantity in quantities}

    print(f"{args.corpus} {args.split}: {len(files)} files, {clip_counts[0]} clips scored")
    print(f"{'quantity':<15}{'cov.':>7}{'rho':>10}{'s.d.':>8}{'nMAE':>8}{'n':>7}{'seeds':>8}")
    for quantity in quantities:
        row = summary[quantity]
        rho = fmt(row["spearman"]["mean"], "%+.3f")
        sd = fmt(row["spearman"]["sd"], "%.3f")
        error = fmt(row["nmae"]["mean"], "%.3f")
        seeds = "%d/%d" % (row["seeds_ranked"], row["seeds"])
        print(f"{quantity:<15}{row['coverage']['mean']:>7.2f}{rho:>10}{sd:>8}{error:>8}"
              f"{row['n_per_seed'][0]:>7}{seeds:>8}")

    out = args.out or os.path.join(args.results, f"decoder_{args.corpus}_{args.split}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"corpus": args.corpus, "split": args.split,
                   "files": [os.path.basename(f) for f in files],
                   "clips_per_file": clip_counts,
                   "thresholds": {"min_claims": MIN_CLAIMS, "min_distinct": MIN_DISTINCT},
                   "quantities": summary}, handle, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
