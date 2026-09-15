# Data

Instrument references, the Libri2Mix mixing manifest and the panel clip lists. No audio is
included. Libri2Mix is built from LibriSpeech and WHAM! with the recipe at
https://github.com/JorisCos/LibriMix; the AMI Meeting Corpus is at
https://groups.inf.ed.ac.uk/ami/corpus/. Both are obtained separately and the file names below
index into them.

## references/

Per-clip values read by the instrument suite. These are the references every system is scored
against. The ten scored quantities are `srmr`, `snr`, `speaking_rate`, `pause_count`, `pause_rate`,
`f0_mean`, `f0_sd`, `jitter`, `shimmer`, `hnr`. The remaining columns, the overlap columns among
them, are carried for reference and are not scored.

`librimix_test.csv`, 6,000 rows, 17 columns, keyed by `filename`. The Libri2Mix test split, 3,000
mixtures and their 3,000 clean twins. A twin carries the `_s1clean` suffix on the mixture's own
name. `snr_db` is the level of the injected noise against the speech. The two F0 columns are read
on the mixture outside the detected overlap windows.

`ami_test.csv`, 3,654 rows, 26 columns, keyed by `filename`. AMI single-distant-microphone clips,
read on the same channel the models hear. `snr_db` here is a within-clip dynamic range, the 90th
over the 10th percentile of frame energy, and is a different quantity from the Libri2Mix column of
that name.

## manifests/

`libri2mix_test_mixtures.csv`, 3,000 rows, the Libri2Mix test mixing manifest as the recipe writes
it. The three path columns are relative to the corpus root, so `mixture_path` reads
`test/mix_clean/<mixture_ID>.wav`.

`panel_clips_librimix.txt`, 839 clip ids, and `panel_clips_ami.txt`, 859 clip ids, one per line.
Each list is the union of the clips drawn across the eight sampling seeds, 120 clips per seed,
which is the set every panel system and the reference decoder are scored on in Table 2. The
Libri2Mix ids are mixtures, with no clean twin among them.

## Scope of the released decoder outputs

`outputs/decoder/librimix_panel839_seed*.json` and `ami_panel859_seed*.json` carry only the clips
these two lists name. The runs they come from covered all 3,000 Libri2Mix test mixtures and all
3,654 AMI test clips; the rest is left out to keep the repository small. Nothing else is filtered,
and `librimix_eval600_seed*.json` is complete at 600 clips, 300 mixtures and their twins.
