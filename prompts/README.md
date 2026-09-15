# Prompts

`prompts.json` holds the five rungs of the elicitation ladder in order. A rung is one elicitation,
and each hands the model strictly more than the one above it, from a free description of the
recording through a request for numbers, the ten quantities named with their units, a template to
complete, and a worked example of the answer. The same five texts go to every system and to both
corpora. One run covers one system, one corpus and one sampling seed. It shuffles the corpus under
that seed, keeps the first 120 clips, truncates each clip to 10 s with no level normalisation, and
asks every rung once per clip, decoding greedily at 160 new tokens. Eight seeds draw their own
samples, so a system contributes 4,800 generations per corpus, and the scorer pools the eight
samples by clip, which is why `n` counts distinct clips carrying a parsed claim rather than
generations.
