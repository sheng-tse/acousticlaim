"""Print the paper's two tables from the scorers' result files, as text and as LaTeX rows."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acousticlaim.quantities import by_key
from acousticlaim.scoring import BAR, EXCLUDED_QUANTITIES

# Quantity, its name as Table 1 prints it and its error case, in the order both tables
# read. The tier of each quantity comes from the quantity table.
ROWS = [("srmr", "SRMR", "(ii)"),
        ("snr", "SNR", "(ii)"),
        ("speaking_rate", "Speaking rate", "(i)"),
        ("pause_count", "Pause count", "(i)"),
        ("pause_rate", "Pause rate", "(i)"),
        ("f0_mean", "F0 mean", "(ii)"),
        ("f0_sd", "F0 s.d.", "(ii)"),
        ("jitter", "Jitter", "(ii)"),
        ("shimmer", "Shimmer", "(ii)"),
        ("hnr", "HNR", "(ii)")]
ORDER = [row[0] for row in ROWS]
# headings of the printed grid, narrow enough to keep the ten columns on one line
HEADING = {"srmr": "SRMR", "snr": "SNR", "speaking_rate": "sp.rate", "pause_count": "p.count",
           "pause_rate": "p.rate", "f0_mean": "F0 mean", "f0_sd": "F0 s.d.", "jitter": "jitter",
           "shimmer": "shimmer", "hnr": "HNR"}

SYSTEMS = ["Qwen2-Audio-7B", "Qwen2.5-Omni-7B", "Granite-Speech-8B", "Audio-Flamingo-3",
           "Gemini-3.8-Flash"]
CORPORA = [("librimix", "Libri2Mix"), ("ami", "AMI")]

# A quantity the decoder withholds on the mixtures is read on the clips' clean twins instead.
TWIN_COVERAGE = 0.10


def load(path):
    """Read one result file, naming it when the scorers have not written it yet."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist, run the two scoring scripts first")
    with open(path) as handle:
        return json.load(handle)


def signed(value):
    return "%+.3f" % value


def table1(results):
    """Table 1, the ten quantities on the Libri2Mix mixtures with the twin rows marked."""
    mixtures = load(os.path.join(results, "decoder_librimix_mixtures.json"))["quantities"]
    twins = load(os.path.join(results, "decoder_librimix_twins.json"))["quantities"]
    rows = []
    for quantity, name, case in ROWS:
        on_twins = mixtures[quantity]["coverage"]["mean"] < TWIN_COVERAGE
        row = (twins if on_twins else mixtures)[quantity]
        rows.append({"name": name, "tier": by_key(quantity).tier, "case": case, "twins": on_twins,
                     "coverage": row["coverage"]["mean"],
                     "rho": row["spearman"]["mean"], "sd": row["spearman"]["sd"],
                     "nmae": row["nmae"]["mean"]})

    print("Table 1  Libri2Mix, readout A, a star where the row is read on the clean twins")
    print(f"{'quantity':<16}{'tier':<12}{'case':<6}{'cov.':>6}{'rho':>9}{'s.d.':>8}{'nMAE':>8}")
    for row in rows:
        name = row["name"] + ("*" if row["twins"] else "")
        print(f"{name:<16}{row['tier']:<12}{row['case']:<6}{row['coverage']:>6.2f}"
              f"{signed(row['rho']):>9}{row['sd']:>8.3f}{row['nmae']:>8.3f}")

    print("\nTable 1, LaTeX rows")
    for row in rows:
        name = row["name"] + ("$^{\\ddagger}$" if row["twins"] else "")
        tier = row["tier"] + ("$^{\\dagger}$" if row["tier"] == "Masked" else "")
        error = ("[$%.3f$]" if row["nmae"] >= 1.0 else "$%.3f$") % row["nmae"]
        print("%s & %s & %s & $%.2f$ & $%s \\pm %.3f$ & %s \\\\"
              % (name, tier, row["case"], row["coverage"], signed(row["rho"]), row["sd"], error))


def panel_grid(cells, corpus):
    """Largest correlation over the rungs per system and quantity, and the cells over the bar."""
    grid, counts = {}, {}
    systems = sorted({cell["system"] for cell in cells})
    for system in systems:
        own = [cell for cell in cells if cell["system"] == system]
        ranked = [cell for cell in own if cell["status"] == "ranked"]
        counts[system] = (sum(1 for cell in ranked if cell["spearman"] > BAR), len(own))
        row = {}
        for quantity in ORDER:
            if quantity in EXCLUDED_QUANTITIES[corpus]:
                row[quantity] = "excl"
                continue
            here = [cell for cell in ranked if cell["quantity"] == quantity]
            if here:
                row[quantity] = max(cell["spearman"] for cell in here)
            elif any(cell["quantity"] == quantity for cell in own):
                row[quantity] = "c"
            else:
                row[quantity] = "--"
        grid[system] = row
    return grid, counts


def ours_row(results, corpus):
    """Our own row, the decoder's emitted text on the same panel clips at its own coverage."""
    summary = load(os.path.join(results, f"decoder_panel_{corpus}.json"))["quantities"]
    row = {}
    for quantity in ORDER:
        if quantity in EXCLUDED_QUANTITIES[corpus]:
            row[quantity] = "excl"
        elif quantity not in summary:
            row[quantity] = "--"
        elif summary[quantity]["seeds_ranked"] == 0:
            row[quantity] = "--" if summary[quantity]["seeds_below_min_claims"] else "c"
        else:
            value = summary[quantity]["spearman"]["mean"]
            row[quantity] = (value, bool(summary[quantity]["seeds_below_min_claims"]))
    return row


def cell_text(value):
    if isinstance(value, tuple):
        return signed(value[0]) + ("*" if value[1] else "")
    return signed(value) if isinstance(value, float) else value


def cell_latex(value):
    if isinstance(value, tuple):
        return signed(value[0]) + ("$^{\\S}$" if value[1] else "")
    if isinstance(value, float):
        return "\\textbf{%s}" % signed(value) if value > BAR else signed(value)
    return value


def table2(results):
    """Table 2, the panel grid over both corpora with our own row beneath each block."""
    blocks, systems = [], set()
    for corpus, name in CORPORA:
        panel = load(os.path.join(results, f"panel_{corpus}.json"))
        grid, counts = panel_grid(panel["cells"], corpus)
        systems |= set(grid)
        blocks.append((corpus, name, grid, counts, ours_row(results, corpus), panel))
    # a system that reaches the claim minimum in no cell of a corpus still gets its row there
    systems = sorted(systems, key=order_of)

    print("\nTable 2  largest correlation over the rungs, a star where a seed of our own row "
          "fell under the claim minimum")
    for corpus, name, grid, counts, ours, panel in blocks:
        cells = panel["cells"]
        ranked = [cell for cell in cells if cell["status"] == "ranked"]
        constant = [cell for cell in cells if cell["status"] == "constant"]
        above = [cell for cell in ranked if cell["spearman"] > BAR]
        print(f"\n{name}: {len(cells)} cells, {len(ranked)} ranked, {len(constant)} under five "
              f"distinct values, {len(above)} over {BAR}")
        header = "".join("%9s" % HEADING[q] for q in ORDER)
        print(f"{'system':<20}{header}{'>0.3':>9}")
        for system in systems:
            row = grid_row(grid, corpus, system)
            line = "".join("%9s" % cell_text(row[q]) for q in ORDER)
            hits, total = counts.get(system, (0, 0))
            print(f"{system:<20}{line}{'%d/%d' % (hits, total):>9}")
        line = "".join("%9s" % cell_text(ours[q]) for q in ORDER)
        print(f"{'AcoustiClaim':<20}{line}{'ref.':>9}")

    print("\nTable 2, LaTeX rows")
    for corpus, name, grid, counts, ours, _ in blocks:
        print("\\multicolumn{12}{@{}l@{}}{\\emph{%s}} \\\\" % name)
        for system in systems:
            row = grid_row(grid, corpus, system)
            values = " & ".join(cell_latex(row[q]) for q in ORDER)
            hits, total = counts.get(system, (0, 0))
            print("%s & %s & %d/%d \\\\" % (system, values, hits, total))
        values = " & ".join(cell_latex(ours[q]) for q in ORDER)
        print("AcoustiClaim & %s & ref. \\\\" % values)


def grid_row(grid, corpus, system):
    if system in grid:
        return grid[system]
    return {q: ("excl" if q in EXCLUDED_QUANTITIES[corpus] else "--") for q in ORDER}


def order_of(system):
    return (SYSTEMS.index(system) if system in SYSTEMS else len(SYSTEMS), system)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results", help="folder the scorers wrote into")
    ap.add_argument("--table", choices=["1", "2", "all"], default="all")
    args = ap.parse_args()
    if args.table in ("1", "all"):
        table1(args.results)
    if args.table in ("2", "all"):
        table2(args.results)


if __name__ == "__main__":
    main()
