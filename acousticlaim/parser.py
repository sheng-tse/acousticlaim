"""Recovery of numeric claims from generated speech descriptions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """One numeric claim recovered from a description."""

    quantity: str
    value: float
    unit: str


class ClaimParser:
    """Reads the thirteen quantities out of free text by quantity name and unit.

    A value is taken verbatim and nothing is rescaled, so a value stated in another
    unit, with no unit, or with the unit before it is left unparsed. Text that states
    no value yields no claims.

        "F0 = 187 Hz"                 -> ("f0_mean", 187.0, "Hz")
        "SNR of approximately 28 dB"  -> ("snr", 28.0, "dB")
        "speaking rate is 7 syl/s"    -> ("speaking_rate", 7.0, "syl/s")
        "The recording contains 3 pauses" -> ("pause_count", 3.0, "")
    """

    # (regex, [(quantity, 1-based group holding the value, unit)]), tried in order.
    # Where two patterns can read the same quantity out of one description, the first
    # match in this order wins.
    PATTERNS = [
        # F0 stated beside its s.d. in one clause: "F0 = 187 Hz (σ = 34 Hz)".
        (r"F0\s*=\s*(\d+\.?\d*)\s*Hz\s*\(?σ\s*=\s*(\d+\.?\d*)\s*Hz", [("f0_mean", 1, "Hz")]),
        # F0 mean: "F0 mean of 96.96 Hz", "mean pitch of 150 Hz", "fundamental
        # frequency mean is 186.69 Hz", "the F0 mean can be estimated at 202.89 Hz".
        (
            r"(?:F0\s*(?:mean\s*)?|(?:mean\s+)?pitch(?:\s+mean)?|fundamental\s+frequency(?:\s+mean)?)\s*(?:=|≈|~|is|of|(?:can\s+be\s+)?(?:estimated|measured)\s+at)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("f0_mean", 1, "Hz")],
        ),
        # F0 listed with a second quantity: "The F0 and speaking rate are 130.16 Hz and
        # 5.221 syl/sec." Binding on Hz picks the F0 value out of the list, and the gap
        # admits a decimal point but not a sentence-ending period.
        (
            r"F0\s+and\b(?:[^.]|\.\d)*?\b(?:are|is)\s+(\d+\.?\d*)\s*Hz",
            [("f0_mean", 1, "Hz")],
        ),
        # SNR, including "Signal-to-Noise Ratio (SNR) is 18.54 dB".
        (
            r"(?:Signal-to-Noise\s+Ratio\s*(?:\(SNR\))?\s*|SNR\s*)(?:=|≈|~|is|of)\s*(?:approximately\s+|estimated at\s+)?(-?\d+\.?\d*)\s*dB",
            [("snr", 1, "dB")],
        ),
        # SNR stated value first: "At 26.15 dB, the signal-to-noise ratio is high",
        # where the word after the quantity name is a band rather than a value.
        (
            r"(-?\d+\.?\d*)\s*dB[,]?(?:[^.]|\.\d)*?(?:signal-to-noise\s+ratio|\bSNR\b)",
            [("snr", 1, "dB")],
        ),
        # HNR, including "Harmonics-to-Noise Ratio (HNR) of 12.59 dB".
        (
            r"(?:Harmonics?-to-Noise\s+Ratio\s*(?:\(HNR\))?\s*|HNR\s*)(?:=|≈|~|is|of)\s*(?:approximately\s+)?(-?\d+\.?\d*)\s*dB",
            [("hnr", 1, "dB")],
        ),
        # Speaking rate. Anchored on the full name so "articulation rate" and
        # "pause rate" cannot match here.
        (
            r"speaking\s+rate\s*(?:=|≈|~|is|of|:)\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?)",
            [("speaking_rate", 1, "syl/s")],
        ),
        # Articulation rate, which excludes pauses where speaking rate includes them.
        (
            r"articulation\s+rate\s*(?:=|≈|~|is|of|:)\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?)",
            [("articulation_rate", 1, "syl/s")],
        ),
        # Value first: "4.1 syl/sec for the articulation rate".
        (
            r"(\d+\.?\d*)\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?\s+for\s+the\s+articulation\s+rate",
            [("articulation_rate", 1, "syl/s")],
        ),
        (
            r"(\d+\.?\d*)\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?\s+for\s+the\s+speaking\s+rate",
            [("speaking_rate", 1, "syl/s")],
        ),
        # Speaking rate listed with another quantity: "The F0 and speaking rate are
        # 130.16 Hz and 5.221 syl/sec." The gap skips the Hz value and refuses to cross
        # "articulation", so a trailing articulation-rate value is never bound here.
        (
            r"speaking\s+rate\b(?:(?!articulation)(?:[^.]|\.\d))*?(\d+\.?\d*)\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?",
            [("speaking_rate", 1, "syl/s")],
        ),
        # Duration. The negative lookahead keeps "7 syl/s" out of the seconds unit.
        (
            r"duration(?:\s+of\s+the\s+(?:speech\s+)?sample)?\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*s(?!yl)",
            [("duration", 1, "s")],
        ),
        (
            r"(?:The\s+)?recording\s+is\s+(\d+\.?\d*)\s*s(?:ec(?:onds?)?)?\s+long",
            [("duration", 1, "s")],
        ),
        # Overlap ratio, unitless on 0 to 1.
        (
            r"overlap\s+ratio(?:\s+of\s+the\s+sample)?\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)",
            [("overlap_ratio", 1, "")],
        ),
        # "a high degree of overlap with a ratio of 0.8261".
        (
            r"overlap[\s,]+with\s+(?:a|an)\s+ratio\s+of\s+(?:approximately\s+)?(\d+\.?\d*)",
            [("overlap_ratio", 1, "")],
        ),
        # Value first: "(0.7825 overlap ratio)".
        (
            r"(\d+\.?\d*)\s+overlap\s+ratio",
            [("overlap_ratio", 1, "")],
        ),
        # F0 s.d., including the shorthand "F0 deviation is X Hz". The F0 mean
        # patterns above cannot reach these because they need a connector or "and"
        # directly after "F0", never the word "deviation".
        (
            r"F0\s+(?:(?:standard\s+)?deviation(?:\s*\(?\s*SD\s*\)?)?|SD)\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("f0_sd", 1, "Hz")],
        ),
        # Split phrasing, "F0 mean is X Hz with a standard deviation of Y Hz". An
        # s.d. in Hz is an F0 s.d. in this corpus.
        (
            r"standard\s+deviation(?:\s*\(?\s*SD\s*\)?)?\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*Hz",
            [("f0_sd", 1, "Hz")],
        ),
        # Pause count, an integer.
        (
            r"pause\s+count\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+)",
            [("pause_count", 1, "")],
        ),
        # "contains a total of 3 pauses", "has 2 pauses", "there are 2 pauses". The
        # verb is required so a bare number beside the word is not taken as a count.
        (
            r"(?:contains?|has|have|with|there\s+(?:are|is))\s+(?:a\s+total\s+of\s+)?(\d+)\s+pauses?\b",
            [("pause_count", 1, "")],
        ),
        # Pause rate, per minute only.
        (
            r"pause\s+rate\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)\s*per\s+min(?:ute)?",
            [("pause_rate", 1, "per min")],
        ),
        # Jitter, including "Jitter local is 2.4784 %" and "jitter local is 1.9686 percent".
        (
            r"jitter\s*(?:\(?\s*(?:local|rap)\s*\)?\s*)?(?:=|≈|~|is|of|\()\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:%|percent)",
            [("jitter", 1, "%")],
        ),
        # Shimmer, including "Shimmer of 13.83 %" and "shimmer is 10.94 percent".
        (
            r"shimmer\s*(?:\(?\s*local\s*\)?\s*)?(?:=|≈|~|is|of|\()\s*(?:approximately\s+)?(\d+\.?\d*)\s*(?:%|percent)",
            [("shimmer", 1, "%")],
        ),
        # SRMR, unitless, including "reverberation score (SRMR) of 9.65".
        (
            r"(?:SRMR|reverberation\s+score\s*(?:\(SRMR\))?)\s*(?:=|≈|~|is|of)\s*(?:approximately\s+)?(\d+\.?\d*)",
            [("srmr", 1, "")],
        ),
    ]

    # Surface forms outside our own house style, rewritten before the patterns run.
    # Only wording changes, never a value, so normalisation cannot invent a claim.
    _EXT_LEXICON = [
        (r"signal[\s-]*to[\s-]*noise(?:\s+ratio)?", "SNR"),
        (r"speech[\s-]*to[\s-]*reverberation(?:\s+modulation\s+energy\s+ratio)?", "SRMR"),
        (r"harmonics?[\s-]*to[\s-]*noise(?:\s+ratio)?", "HNR"),
        (r"(?:average|mean|median)\s+pitch", "F0 mean"),
        (r"pitch\s+(?:standard\s+deviation|variability|sd)", "F0 SD"),
        (r"\bpitch\b", "F0 mean"),
        (r"(?:speech|speaking|articulation)\s+(?:rate|speed)", "speaking rate"),
        (r"(?:number\s+of\s+)?(?:silent\s+)?(?:gaps|silences)", "pauses"),
    ]
    _EXT_UNITS = [
        (r"\bdecibels?\b", "dB"), (r"\bhertz\b", "Hz"),
        (r"\bsyllables?\s*(?:/|per)\s*(?:s|sec|second)\b", "syllables per second"),
        (r"\bsyl\s*/\s*s\b", "syllables per second"),
        (r"\bpercent\b", "%"),
    ]

    @classmethod
    def normalize_external(cls, text: str) -> str:
        """Rewrite out-of-family phrasing into the name-connector-value-unit form."""
        t = text
        t = re.sub(r"^[ \t]*[\*\-•–—\+]+[ \t]*", "", t, flags=re.MULTILINE)  # list bullets
        t = re.sub(r"\*\*|__|`", "", t)  # markdown emphasis
        for pat, rep in cls._EXT_LEXICON:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        # An expanded name followed by its own acronym becomes "HNR (HNR)" once the
        # lexicon has run, and the parenthetical then breaks the adjacency the patterns
        # need. Collapse an acronym immediately followed by itself in brackets.
        t = re.sub(r"\b([A-Za-z][A-Za-z0-9]*)\s*\(\s*\1\s*\)", r"\1", t, flags=re.IGNORECASE)
        for pat, rep in cls._EXT_UNITS:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        # A separator becomes "is" only where a number follows, so prose colons stay.
        t = re.sub(r"\s*[:–—]\s*(?=[<>~≈]?\s*-?\d)", " is ", t)
        t = re.sub(r"(?<=[A-Za-z0-9)])\s+-\s+(?=-?\d)", " is ", t)
        # Bare "N pauses" and "pauses is N" become the phrasing the patterns expect.
        t = re.sub(r"\bpauses\s+is\s+(\d+)", r"contains \1 pauses", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(?:roughly|about|around|approximately)\s+(\d+)\s+pauses",
                   r"contains \1 pauses", t, flags=re.IGNORECASE)
        return t

    def parse(self, text: str, normalize: bool = False) -> list[Claim]:
        """Extract every numeric claim from a description, at most one per quantity.

        Set normalize for text written by a system whose surface form is not ours.
        """
        if normalize:
            text = self.normalize_external(text)

        claims: list[Claim] = []
        for pattern, extractions in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                for quantity, group_idx, unit in extractions:
                    try:
                        value = float(match.group(group_idx))
                    except (ValueError, IndexError):
                        continue
                    claims.append(Claim(quantity=quantity, value=value, unit=unit))

        seen = set()
        unique: list[Claim] = []
        for claim in claims:
            if claim.quantity not in seen:
                seen.add(claim.quantity)
                unique.append(claim)
        return unique

    def parse_external(self, text: str) -> list[Claim]:
        """Parse a description written by a system other than ours."""
        return self.parse(text, normalize=True)

