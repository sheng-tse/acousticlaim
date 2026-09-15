"""Print the numeric claims a description states."""

from __future__ import annotations

import argparse

from acousticlaim.parser import ClaimParser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", required=True, help="description to parse")
    ap.add_argument("--external", action="store_true",
                    help="normalise phrasing first, for text not written by our own decoder")
    args = ap.parse_args()
    for claim in ClaimParser().parse(args.text, normalize=args.external):
        print("%-18s %g %s" % (claim.quantity, claim.value, claim.unit))


if __name__ == "__main__":
    main()
