"""Metrics, admission thresholds and reference loading for the benchmark."""

from __future__ import annotations

import csv
import math

import numpy as np
from scipy.stats import spearmanr

from .quantities import SCORED

# A cell is scored once a system has stated this many parsed claims for a quantity, and ranked
# once those claims take at least this many distinct values. Below the second threshold a rank
# correlation measures tie handling rather than the ordering.
MIN_CLAIMS = 25
MIN_DISTINCT = 5

# A ranked cell tracks the reference above this Spearman correlation.
BAR = 0.3

# References that measure something other than the quantity the prompt asks for. No level was
# injected on AMI, so its SNR reference is a within-clip dynamic range, and its two pause
# references read zero on nine clips in ten.
EXCLUDED_QUANTITIES = {"librimix": (), "ami": ("snr", "pause_count", "pause_rate")}

# The released files name the first corpus librimix and the quantity table names it libri2mix.
CORPUS_KEYS = {"librimix": "libri2mix", "ami": "ami"}

# The clean twin of a Libri2Mix mixture carries this suffix.
TWIN_SUFFIX = "_s1clean"
SPLITS = ("pooled", "mixtures", "twins")


def reference_columns(corpus):
    """Reference CSV column of every quantity scored on this corpus."""
    excluded = EXCLUDED_QUANTITIES[corpus]
    columns = {}
    for quantity in SCORED:
        column = quantity.column(CORPUS_KEYS[corpus])
        if column and quantity.key not in excluded:
            columns[quantity.key] = column
    return columns


def scored_quantities(corpus):
    """Quantity names scored on this corpus, in table order."""
    return tuple(reference_columns(corpus))


def load_reference(path, corpus):
    """Read a reference CSV into {clip: {quantity: value}}."""
    columns = reference_columns(corpus)
    table = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        clip_field = reader.fieldnames[0]
        missing = [column for column in columns.values() if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} has no column {', '.join(sorted(missing))}")
        for row in reader:
            values = {}
            for quantity, column in columns.items():
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values[quantity] = value
            table[strip_extension(row[clip_field])] = values
    return table


def strip_extension(clip):
    return str(clip).replace(".wav", "").replace(".pt", "")


def clip_id(record):
    """Clip a generation record was produced from."""
    clip = record.get("clip") or record.get("stem") or record.get("filename") or ""
    return strip_extension(clip)


def generated_text(record):
    return record.get("text") or record.get("generated_clean") or record.get("generated") or ""


def in_split(clip, split):
    """Whether a clip belongs to the requested half of the Libri2Mix evaluation set."""
    if split == "pooled":
        return True
    twin = clip.endswith(TWIN_SUFFIX)
    return twin if split == "twins" else not twin


def parse_claims(parser, text, quantities, external=False):
    """Value the parser reads for each quantity, the first statement winning.

    External prose is normalised into the family's own phrasing first, so a panel system is
    not scored with an instrument tuned to our own wording.
    """
    claims = {}
    for claim in (parser.parse_external(text) if external else parser.parse(text)):
        if claim.quantity in quantities:
            claims.setdefault(claim.quantity, claim.value)
    return claims


def spearman(stated, reference):
    """Spearman correlation, or None where it does not exist."""
    stated = np.asarray(stated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(stated) < 3 or np.std(stated) == 0 or np.std(reference) == 0:
        return None
    value = spearmanr(stated, reference).correlation
    return float(value) if np.isfinite(value) else None


def constant_floor(reference):
    """Absolute error of the constant predictor, the median of the reference over these rows."""
    reference = np.asarray(reference, dtype=float)
    return float(np.mean(np.abs(reference - np.median(reference))))


def nmae(stated, reference, floor=None):
    """Mean absolute error over the constant-predictor floor, so 1.0 is that floor."""
    stated = np.asarray(stated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if floor is None:
        floor = constant_floor(reference)
    if not floor > 0:
        return None
    return float(np.mean(np.abs(stated - reference)) / floor)


def score_cell(stated, reference, n_clips=None):
    """Score the values one system stated for one quantity against the instrument.

    status is few under MIN_CLAIMS, constant under MIN_DISTINCT, undefined where the correlation
    does not exist, and ranked otherwise. nMAE is reported whenever a value was stated.
    """
    stated = np.asarray(stated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    cell = {"n": int(len(stated)), "distinct": 0, "distinct_reference": 0, "floor": None,
            "spearman": None, "nmae": None, "status": "few"}
    if n_clips:
        cell["coverage"] = len(stated) / n_clips
    if len(stated) == 0:
        return cell
    cell["distinct"] = int(len(np.unique(np.round(stated, 6))))
    cell["distinct_reference"] = int(len(np.unique(np.round(reference, 9))))
    cell["floor"] = constant_floor(reference)
    cell["nmae"] = nmae(stated, reference, cell["floor"])
    if len(stated) < MIN_CLAIMS:
        return cell
    if cell["distinct"] < MIN_DISTINCT:
        cell["status"] = "constant"
        return cell
    cell["spearman"] = spearman(stated, reference)
    cell["status"] = "ranked" if cell["spearman"] is not None else "undefined"
    return cell


def mean_sd(values):
    """Mean and sample standard deviation over the values that exist."""
    kept = [value for value in values if value is not None and math.isfinite(value)]
    if not kept:
        return {"mean": None, "sd": None, "k": 0}
    return {"mean": float(np.mean(kept)),
            "sd": float(np.std(kept, ddof=1)) if len(kept) > 1 else 0.0,
            "k": len(kept)}
