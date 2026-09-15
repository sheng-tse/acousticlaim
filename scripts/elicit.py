"""Run the five prompt rungs over one clip sample with one open-weight audio language model."""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPO_ROOT / "prompts" / "prompts.json"
CORPORA = ("librimix", "ami")


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


def write_temp_wav(x, sr):
    import soundfile as sf

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(handle.name, x, sr)
    return handle.name


def decode_new(processor, out, n_in):
    """Decode the generated continuation, dropping the n_in prompt tokens."""
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]


def load_qwen2_audio(max_new_tokens):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto").eval()

    def generate(x, sr, prompt):
        # the chat template needs an audio placeholder; the waveform goes to the processor
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "a.wav"},
                                             {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[x], sampling_rate=sr,
                           return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return decode_new(processor, out, inputs.input_ids.shape[1])

    return generate, model_id


def load_qwen25_omni(max_new_tokens):
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model_id = "Qwen/Qwen2.5-Omni-7B"
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto").eval()

    def generate(x, sr, prompt):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "a.wav"},
                                             {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[x], sampling_rate=sr,
                           return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 return_audio=False, thinker_max_new_tokens=max_new_tokens)
        # the model returns a pair when it also speaks
        out = out[0] if isinstance(out, (tuple, list)) else out
        return decode_new(processor, out, inputs.input_ids.shape[1])

    return generate, model_id


def load_granite_speech(max_new_tokens):
    import torch
    from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

    model_id = "ibm-granite/granite-speech-3.3-8b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16).to("cuda").eval()

    def generate(x, sr, prompt):
        # the audio placeholder goes in the chat text and the waveform alongside it
        chat = [{"role": "user", "content": "<|audio|>" + prompt}]
        text = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        wav = torch.tensor(x).unsqueeze(0)
        inputs = processor(text, wav, return_tensors="pt").to("cuda")
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return decode_new(processor, out, n_in)

    return generate, model_id


def load_audio_flamingo3(max_new_tokens):
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    model_id = "nvidia/audio-flamingo-3-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16).to("cuda").eval()

    def generate(x, sr, prompt):
        path = write_temp_wav(x, sr)
        try:
            conv = [{"role": "user", "content": [{"type": "audio", "path": path},
                                                 {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                conv, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to("cuda")
            # the processor emits float32 features and the audio tower runs in the model dtype
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    inputs[key] = value.to(model.dtype)
            n_in = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            os.unlink(path)
        return decode_new(processor, out, n_in)

    return generate, model_id


SYSTEMS = {
    "Qwen2-Audio-7B": load_qwen2_audio,
    "Qwen2.5-Omni-7B": load_qwen25_omni,
    "Granite-Speech-8B": load_granite_speech,
    "Audio-Flamingo-3": load_audio_flamingo3,
}


def run_rung(generate, prompt, clips, seconds):
    """Generate one rung over every clip, returning the texts and the clips that failed."""
    import torch

    generations, errors = [], []
    for path in clips:
        text = None
        for attempt in (0, 1):
            try:
                x, sr = load_audio(path, seconds)
                text = generate(x, sr, prompt)
                break
            except Exception as exc:
                # an out-of-memory failure retries once on a cleared cache, then drops the clip
                if attempt == 0 and "memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    continue
                errors.append("%s: %s" % (type(exc).__name__, str(exc)[:120]))
                break
        if text is None:
            continue
        generations.append({"stem": path.stem, "text": text})
    return generations, errors


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--corpus", default="librimix", choices=CORPORA)
    ap.add_argument("--audio-dir", default=None,
                    help="directory of corpus wav files (default: data/audio/<corpus>)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: %(default)s)")
    ap.add_argument("--clips", type=int, default=120,
                    help="clips drawn per seed (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output json (default: outputs/panel/<system>_<corpus>_seed<seed>.json)")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS),
                    help="prompt ladder json (default: %(default)s)")
    ap.add_argument("--max-new-tokens", type=int, default=160,
                    help="greedy decoding budget (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="clip truncation in seconds (default: %(default)s)")
    return ap.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir or ("data/audio/" + args.corpus))
    rungs = json.loads(Path(args.prompts).read_text())["rungs"]
    clips = sample_clips(audio_dir, args.seed, args.clips)

    slug = args.system.lower()
    out_path = Path(args.out) if args.out else Path(
        "outputs/panel/%s_%s_seed%d.json" % (slug, args.corpus, args.seed))

    generate, model_id = SYSTEMS[args.system](args.max_new_tokens)

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
        "system": args.system,
        "model_id": model_id,
        "corpus": args.corpus,
        "seed": args.seed,
        "n_clips": len(clips),
        "max_new_tokens": args.max_new_tokens,
        "results": {args.system: rung_records},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote " + str(out_path))


if __name__ == "__main__":
    main()
