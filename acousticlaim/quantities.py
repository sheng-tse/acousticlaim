"""The quantity table: what the parser reads and what the scorer scores.

Ten quantities are scored. Overlap ratio, articulation rate and duration are parsed
but never scored, overlap ratio because it echoes an input channel and the other two
because no table reports them.
"""

from __future__ import annotations

from dataclasses import dataclass

CORPORA = ("libri2mix", "ami")


@dataclass(frozen=True)
class Quantity:
    """One quantity: how it is named, where its reference is read, whether it is scored."""

    key: str
    name: str
    unit: str
    # Exact is the signal the model hears, Measurable a mixing parameter, Stem-only the
    # clean stem it never hears, Masked the mixture outside the detected overlap windows.
    tier: str | None
    aliases: tuple[str, ...]
    libri2mix_column: str | None
    ami_column: str | None
    scored: bool

    def column(self, corpus: str) -> str | None:
        """Column holding this quantity's reference in the corpus feature table."""
        if corpus not in CORPORA:
            raise ValueError(f"unknown corpus {corpus!r}, expected one of {CORPORA}")
        return self.libri2mix_column if corpus == "libri2mix" else self.ami_column


# Order is the order the paper's two tables print. Tier is the tier on the Libri2Mix
# mixture; on AMI every reference is read on the distant channel the models hear.
QUANTITIES: tuple[Quantity, ...] = (
    Quantity(
        key="srmr",
        name="SRMR",
        unit="",
        tier="Exact",
        aliases=(
            "speech-to-reverberation modulation energy ratio",
            "reverberation score",
        ),
        libri2mix_column="srmr",
        ami_column="srmr",
        scored=True,
    ),
    Quantity(
        key="snr",
        name="SNR",
        unit="dB",
        tier="Measurable",
        aliases=("signal-to-noise ratio",),
        libri2mix_column="snr_db",
        ami_column="snr_db",
        scored=True,
    ),
    Quantity(
        key="speaking_rate",
        name="speaking rate",
        unit="syl/s",
        tier="Stem-only",
        aliases=("speech rate",),
        libri2mix_column="praat_speaking_rate_syl_sec",
        ami_column="praat_speaking_rate_syl_sec",
        scored=True,
    ),
    Quantity(
        key="pause_count",
        name="pause count",
        unit="",
        tier="Stem-only",
        aliases=("number of pauses", "pauses"),
        libri2mix_column="praat_pause_count",
        ami_column="praat_pause_count",
        scored=True,
    ),
    Quantity(
        key="pause_rate",
        name="pause rate",
        unit="per min",
        tier="Stem-only",
        aliases=("pauses per minute",),
        libri2mix_column="praat_pause_rate_per_min",
        ami_column="praat_pause_rate_per_min",
        scored=True,
    ),
    Quantity(
        key="f0_mean",
        name="F0 mean",
        unit="Hz",
        tier="Masked",
        aliases=("fundamental frequency mean", "mean pitch", "pitch"),
        libri2mix_column="f0_mean_hz",
        ami_column="f0_mean_hz",
        scored=True,
    ),
    Quantity(
        key="f0_sd",
        name="F0 s.d.",
        unit="Hz",
        tier="Masked",
        aliases=(
            "f0 sd",
            "f0 standard deviation",
            "fundamental frequency standard deviation",
            "pitch standard deviation",
        ),
        libri2mix_column="f0_sd_hz",
        ami_column="f0_sd_hz",
        scored=True,
    ),
    Quantity(
        key="jitter",
        name="jitter",
        unit="%",
        tier="Stem-only",
        aliases=("jitter local",),
        libri2mix_column="jitter_local_pct",
        ami_column="jitter_local_pct",
        scored=True,
    ),
    Quantity(
        key="shimmer",
        name="shimmer",
        unit="%",
        tier="Stem-only",
        aliases=("shimmer local",),
        libri2mix_column="shimmer",
        ami_column="shimmer_pct",
        scored=True,
    ),
    Quantity(
        key="hnr",
        name="HNR",
        unit="dB",
        tier="Stem-only",
        aliases=("harmonics-to-noise ratio", "harmonic-to-noise ratio"),
        libri2mix_column="hnr",
        ami_column="hnr_db",
        scored=True,
    ),
    Quantity(
        key="overlap_ratio",
        name="overlap ratio",
        unit="",
        tier=None,
        aliases=(),
        libri2mix_column="overlap_ratio",
        ami_column="overlap_ratio",
        scored=False,
    ),
    Quantity(
        key="articulation_rate",
        name="articulation rate",
        unit="syl/s",
        tier=None,
        aliases=(),
        libri2mix_column="praat_articulation_rate_syl_sec",
        ami_column="praat_articulation_rate_syl_sec",
        scored=False,
    ),
    Quantity(
        key="duration",
        name="duration",
        unit="s",
        tier=None,
        aliases=("recording length",),
        libri2mix_column=None,
        ami_column="duration_sec",
        scored=False,
    ),
)

SCORED: tuple[Quantity, ...] = tuple(q for q in QUANTITIES if q.scored)
SCORED_KEYS: tuple[str, ...] = tuple(q.key for q in SCORED)
KEYS: tuple[str, ...] = tuple(q.key for q in QUANTITIES)

_BY_KEY = {q.key: q for q in QUANTITIES}
_BY_ALIAS = {
    alias: q.key
    for q in QUANTITIES
    for alias in (q.key, q.name.lower()) + q.aliases
}


def by_key(key: str) -> Quantity:
    """Quantity with this key."""
    return _BY_KEY[key]


def resolve(name: str) -> str | None:
    """Key of the quantity named by its key, display name or an alias, else None."""
    return _BY_ALIAS.get(" ".join(name.lower().split()))
