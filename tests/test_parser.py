"""Parser cases, taken from the house-style forms and from panel generations."""

import pytest

from acousticlaim import quantities
from acousticlaim.parser import ClaimParser

PARSER = ClaimParser()

# One Gemini-3.8-Flash answer to the quantities-named rung, verbatim.
GEMINI_BULLETS = (
    "- **Signal-to-Noise Ratio (SNR)**: 23.49 dB\n"
    "- **Speech-to-Reverberation Modulation Energy Ratio (SRMR)**: 7.73\n"
    "- **Fundamental Frequency Mean (F0 Mean)**: 153.30 Hz\n"
    "- **Fundamental Frequency Standard Deviation (F0 SD)**: 35.80 Hz\n"
    "- **Speaking Rate**: 2.94 syllables per second\n"
    "- **Number of Pauses**: 1\n"
    "- **Pause Rate**: 11.76 pauses per minute\n"
    "- **Jitter**: 0.44%\n"
    "- **Shimmer**: 3.52%\n"
    "- **Harmonics-to-Noise Ratio**: 15.65 dB"
)

# One Qwen2.5-Omni-7B answer to the template rung, verbatim.
OMNI_TEMPLATE = (
    "The SNR is 20 dB. The SRMR is 0.05. The F0 mean is 120 Hz. The F0 SD is 5 Hz. "
    "The speaking rate is 3.5 syllables per second. The recording contains 5 pauses. "
    "The pause rate is 1 per minute. The jitter is 2 percent. The shimmer is 3 percent. "
    "The HNR is 30 dB."
)

# One Audio-Flamingo-3 answer to the worked-example rung, verbatim: the example's own
# numbers returned, with HNR misspelt.
AF3_FEWSHOT = (
    "The SNR is 12.4 dB. The SRMR is 6.8. The F0 mean is 138.2 Hz. The F0 SD is 21.5 Hz. "
    "The speaking rate is 3.9 syllables per second. The recording contains 5 pauses. "
    "The pause rate is 18.7 per minute. The jitter is 1.3 percent. The shimmer is 4.6 percent. "
    "The HHR is 14.2 dB."
)


def parsed(text, external=False):
    claims = PARSER.parse_external(text) if external else PARSER.parse(text)
    return {c.quantity: c.value for c in claims}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("F0 = 187 Hz", ("f0_mean", 187.0, "Hz")),
        ("SNR ≈ 28 dB", ("snr", 28.0, "dB")),
        ("SNR of approximately 28 dB", ("snr", 28.0, "dB")),
        ("speaking rate: 7 syl/s", ("speaking_rate", 7.0, "syl/s")),
        ("Harmonics-to-Noise Ratio (HNR) of 12.59 dB", ("hnr", 12.59, "dB")),
        ("reverberation score (SRMR) of 9.65", ("srmr", 9.65, "")),
        ("Jitter local is 2.4784 %", ("jitter", 2.4784, "%")),
        ("The pause count is 4", ("pause_count", 4.0, "")),
    ],
)
def test_single_claim_forms(text, expected):
    claims = PARSER.parse(text)
    assert len(claims) == 1
    assert (claims[0].quantity, claims[0].value, claims[0].unit) == expected


def test_template_answer_yields_all_ten_scored():
    assert parsed(OMNI_TEMPLATE) == {
        "snr": 20.0,
        "srmr": 0.05,
        "f0_mean": 120.0,
        "f0_sd": 5.0,
        "speaking_rate": 3.5,
        "pause_count": 5.0,
        "pause_rate": 1.0,
        "jitter": 2.0,
        "shimmer": 3.0,
        "hnr": 30.0,
    }


def test_bullets_and_expanded_names_need_normalisation():
    assert parsed(GEMINI_BULLETS) == {"speaking_rate": 2.94}
    assert parsed(GEMINI_BULLETS, external=True) == {
        "snr": 23.49,
        "srmr": 7.73,
        "speaking_rate": 2.94,
        "pause_count": 1.0,
        "jitter": 0.44,
        "shimmer": 3.52,
        "hnr": 15.65,
    }


def test_name_restated_in_brackets_is_dropped():
    # The two F0 lines of GEMINI_BULLETS: the bracketed restatement separates the name
    # from the value, so neither is read.
    assert parsed("- **Fundamental Frequency Mean (F0 Mean)**: 153.30 Hz", external=True) == {}


def test_misspelt_quantity_name_drops_that_claim():
    claims = parsed(AF3_FEWSHOT)
    assert "hnr" not in claims
    assert claims["snr"] == 12.4


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I cannot provide the requested measurements.",
        "She succeeded in opening the window.",
        "The recording is of a male speaker in a quiet room.",
    ],
)
def test_a_description_that_states_no_value_returns_nothing(text):
    assert PARSER.parse_external(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "The F0 mean is 45.3 semitones.",
        "The F0 mean is 0.1382 kHz.",
        "The F0 mean is 138.2.",
        "The SNR is 12,4 dB.",
        "The SNR in dB is 12.4.",
        "The speaking rate is 234 syllables per minute.",
        "The jitter is 0.013.",
        "The pause rate is 18.7 pauses per minute.",
        "The recording contains 5.0 pauses.",
        '{"SNR": "12.4 dB"}',
    ],
)
def test_values_outside_the_unit_policy_are_dropped(text):
    assert PARSER.parse_external(text) == []


def test_negative_and_approximate_values_are_read():
    assert parsed("The SNR is -3.5 dB.") == {"snr": -3.5}
    assert parsed("SNR of approximately 12.4 dB") == {"snr": 12.4}


def test_srmr_ignores_a_spurious_unit():
    assert parsed("The SRMR is 6.8 dB.") == {"srmr": 6.8}


def test_value_stated_before_the_quantity_name():
    assert parsed("At 26.15 dB, the signal-to-noise ratio SNR is high.") == {"snr": 26.15}


def test_two_quantities_in_one_sentence_bind_by_unit():
    assert parsed("The F0 and speaking rate are 130.16 Hz and 5.221 syl/sec.") == {
        "f0_mean": 130.16,
        "speaking_rate": 5.221,
    }


def test_at_most_one_claim_per_quantity():
    assert parsed("The SNR is 12.4 dB. The SNR is 30.0 dB.") == {"snr": 12.4}


def test_unscored_quantities_are_parsed_but_not_scored():
    claims = parsed(
        "The overlap ratio is 0.7528. The articulation rate is 5.10 syl/sec. "
        "The recording is 10.0 s long."
    )
    assert claims == {"overlap_ratio": 0.7528, "articulation_rate": 5.1, "duration": 10.0}
    assert not any(quantities.by_key(k).scored for k in claims)


def test_patterns_only_name_quantities_in_the_table():
    named = {q for _pattern, extractions in ClaimParser.PATTERNS for q, _g, _u in extractions}
    assert named <= set(quantities.KEYS)
    assert set(quantities.SCORED_KEYS) <= named
