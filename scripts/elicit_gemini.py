"""Run the five prompt rungs over one clip sample with a Gemini model through its API."""

import argparse
import io
import json
import os
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPO_ROOT / "prompts" / "prompts.json"
DEFAULT_MODEL = "gemini-3.8-flash"
CORPORA = ("librimix", "ami")
SYSTEM_NAMES = {"gemini-3.8-flash": "Gemini-3.8-Flash"}
MAX_ATTEMPTS = 6
TOKEN_FIELDS = ("prompt_token_count", "candidates_token_count",
                "thoughts_token_count", "total_token_count")
CALL_FIELDS = ("n_calls", "n_retries", "n_429", "n_empty", "n_gave_up")


def sample_clips(audio_dir, seed, n_clips):
    """Shuffle the corpus once under the seed and keep the first n_clips."""
    wavs = sorted(Path(audio_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit("no .wav files under " + str(audio_dir))
    random.Random(seed).shuffle(wavs)
    return wavs[:n_clips]


def load_audio(path, seconds):
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return x[: int(seconds * sr)], sr


def load_gemini(model_id, max_new_tokens):
    import soundfile as sf
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment")
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=max_new_tokens,
        thinking_config=types.ThinkingConfig(thinking_level="low"))
    usage = dict.fromkeys(TOKEN_FIELDS + CALL_FIELDS, 0)

    def generate(x, sr, prompt):
        # the clip goes up as an in-memory 16-bit wav, so no temporary file is written
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
        part = types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav")
        last = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=model_id, contents=[part, prompt], config=config)
            except Exception as exc:
                last = exc
                rate_limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                usage["n_retries"] += 1
                if rate_limited:
                    usage["n_429"] += 1
                time.sleep(min(90.0, (5.0 if rate_limited else 2.0) * (2 ** attempt)))
                continue
            metadata = getattr(response, "usage_metadata", None)
            if metadata is not None:
                for field in TOKEN_FIELDS:
                    usage[field] += int(getattr(metadata, field, 0) or 0)
            usage["n_calls"] += 1
            try:
                text = response.text
            except Exception:
                # the accessor raises when the response carries no text part
                text = None
            # a blocked or empty response counts as a generation that states nothing
            if not text:
                usage["n_empty"] += 1
            return text or ""
        usage["n_gave_up"] += 1
        raise RuntimeError("%s gave up after %d attempts: %s: %s"
                           % (model_id, MAX_ATTEMPTS, type(last).__name__, str(last)[:120]))

    return generate, usage


def run_rung(generate, prompt, clips, seconds):
    """Generate one rung over every clip, returning the texts and the clips that failed."""
    generations, errors = [], []
    for path in clips:
        try:
            x, sr = load_audio(path, seconds)
            text = generate(x, sr, prompt)
        except Exception as exc:
            errors.append("%s: %s" % (type(exc).__name__, str(exc)[:120]))
            continue
        generations.append({"stem": path.stem, "text": text})
    return generations, errors


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Gemini model id (default: %(default)s)")
    ap.add_argument("--corpus", default="librimix", choices=CORPORA)
    ap.add_argument("--audio-dir", default=None,
                    help="directory of corpus wav files (default: data/audio/<corpus>)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: %(default)s)")
    ap.add_argument("--clips", type=int, default=120,
                    help="clips drawn per seed (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output json (default: outputs/panel/<model>_<corpus>_seed<seed>.json)")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS),
                    help="prompt ladder json (default: %(default)s)")
    ap.add_argument("--max-new-tokens", type=int, default=160,
                    help="output token budget (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="clip truncation in seconds (default: %(default)s)")
    return ap.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir or ("data/audio/" + args.corpus))
    system = SYSTEM_NAMES.get(args.model, args.model)
    rungs = json.loads(Path(args.prompts).read_text())["rungs"]
    clips = sample_clips(audio_dir, args.seed, args.clips)

    out_path = Path(args.out) if args.out else Path(
        "outputs/panel/%s_%s_seed%d.json" % (args.model, args.corpus, args.seed))

    generate, usage = load_gemini(args.model, args.max_new_tokens)

    rung_records = {}
    for rung in rungs:
        generations, errors = run_rung(generate, rung["prompt"], clips, args.seconds)
        rung_records[rung["name"]] = {
            "prompt": rung["prompt"],
            "n_clips": len(clips),
            "n_scored": len(generations),
            "n_errors": len(errors),
            "error_frac": len(errors) / len(clips),
            "errors_sample": errors[:5],
            "generations": generations,
        }
        print("%s %d/%d generations, %d errors"
              % (rung["name"], len(generations), len(clips), len(errors)), flush=True)

    record = {
        "system": system,
        "model_id": args.model,
        "corpus": args.corpus,
        "seed": args.seed,
        "n_clips": len(clips),
        "max_new_tokens": args.max_new_tokens,
        "results": {system: rung_records},
        "usage": {args.model: usage},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote " + str(out_path))


if __name__ == "__main__":
    main()
