"""Probe what a checkpoint's PaliGemma backbone still knows about our scenes.

Runs the vision encoder + backbone of a pi0/pi0.5 checkpoint as a plain PaliGemma, with PaliGemma's own prompt
formats ("caption en", "detect X", "answer en ..."), on the frames of a probe set. The action expert is not used.
For every frame it generates a caption and one box per "detect" object, and scores yes/no and multiple-choice
questions by the log-likelihood of each answer (robust to a backbone that no longer writes clean text).

One checkpoint per process (one model in memory at a time), then compare two runs:

    uv run scripts/probe_backbone.py run --checkpoint gs://openpi-assets/checkpoints/pi05_base/params \
        --probe-set ../../analysis/backbone_probe/probe_set.json --out ../../analysis/backbone_probe/results --name base
    uv run scripts/probe_backbone.py run --checkpoint hf://FilippoGorini/pi05_kinova_clutter_rtc/30000 \
        --probe-set ../../analysis/backbone_probe/probe_set.json --out ../../analysis/backbone_probe/results --name 30k
    uv run scripts/probe_backbone.py compare --runs base 30k --out ../../analysis/backbone_probe/results

`--dummy` builds a tiny random model instead of loading a checkpoint, to test the code path on a small GPU.
"""

import dataclasses
import json
import logging
import pathlib

import numpy as np
from PIL import Image
from PIL import ImageDraw
import tyro

TEXT_LEN = 64  # fixed text length (prompt + answer), so each function compiles once
LOC0 = 256000  # PaliGemma <loc0000>; <loc0000>..<loc1023> are ids 256000..257023
EOS = 1
IMG_SIZE = 224
COLORS = ["#ff3030", "#30c0ff", "#ffd000", "#40ff40", "#ff40ff", "#ff9020"]


@dataclasses.dataclass
class Run:
    """Probe one checkpoint."""

    # Checkpoint directory: local path or URL (gs://..., hf://...). Must contain params/ (or be the params dir).
    checkpoint: str
    # Probe set JSON (frame paths are relative to it).
    probe_set: str
    # Name of this run; results go to <out>/<name>.
    name: str
    # Training config whose model class is used to load the params.
    config: str = "pi05_kinova_clutter_rtc"
    out: str = "probe_results"
    # Max tokens generated for captions and detections.
    max_new: int = 32
    # Tiny random model instead of the checkpoint (code-path test only).
    dummy: bool = False


@dataclasses.dataclass
class Compare:
    """Compare two runs side by side."""

    # Two run names (subfolders of --out), e.g. base 30k.
    runs: tuple[str, str]
    out: str = "probe_results"


def _load_model(args: Run):
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.shared import download
    from openpi.training import config as _config

    model_config = _config.get_config(args.config).model
    if args.dummy:
        model_config = dataclasses.replace(model_config, paligemma_variant="dummy", action_expert_variant="dummy")
        return model_config.create(jax.random.key(0))
    path = download.maybe_download(args.checkpoint, ignore_patterns=("train_state/*",))
    params_dir = path if path.name == "params" else path / "params"
    return model_config.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))


def _make_fns(model):
    """Jitted (image -> image tokens) and (tokens -> per-position argmax, log p(next text token))."""
    from flax import nnx
    import jax
    import jax.numpy as jnp

    from openpi.models.pi0 import make_attn_mask

    graphdef, state = nnx.split(model)

    @jax.jit
    def encode(state, image):
        tokens, _ = nnx.merge(graphdef, state).PaliGemma.img(image, train=False)
        return tokens

    @jax.jit
    def forward(state, img_tokens, text, text_mask, text_ar):
        llm = nnx.merge(graphdef, state).PaliGemma.llm
        txt_emb = llm(text, method="embed")
        n_img = img_tokens.shape[1]
        emb = jnp.concatenate([img_tokens.astype(txt_emb.dtype), txt_emb], axis=1)
        input_mask = jnp.concatenate([jnp.ones((1, n_img), bool), text_mask], axis=1)
        # Prefix-LM like PaliGemma: image + prompt attend to each other, answer tokens are causal.
        ar = jnp.concatenate([jnp.zeros((1, n_img), bool), text_ar], axis=1)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (out, _), _ = llm([emb, None], mask=make_attn_mask(input_mask, ar), positions=positions)
        # Tied LM head, same as gemma.Embedder.decode (not reachable through the module on the committed branch).
        table = state["PaliGemma"]["llm"]["embedder"]["input_embedding"].value
        logp = jax.nn.log_softmax(jnp.dot(out[:, n_img:], table.T).astype(jnp.float32), axis=-1)
        next_logp = jnp.take_along_axis(logp[:, :-1], text[:, 1:, None], axis=-1)[..., 0]
        return jnp.argmax(logp, axis=-1), next_logp

    return (lambda image: encode(state, image)), (lambda *a: forward(state, *a))


def _pack(ids: list[int], n_prompt: int):
    assert len(ids) <= TEXT_LEN, f"prompt + answer is {len(ids)} tokens, TEXT_LEN is {TEXT_LEN}"
    text = np.zeros((1, TEXT_LEN), np.int32)
    text[0, : len(ids)] = ids
    mask = np.zeros((1, TEXT_LEN), bool)
    mask[0, : len(ids)] = True
    ar = np.zeros((1, TEXT_LEN), bool)
    ar[0, n_prompt : len(ids)] = True
    return text, mask, ar


def _preprocess(path: pathlib.Path):
    from openpi.shared import image_tools

    img = np.asarray(Image.open(path).convert("RGB"))
    small = np.asarray(image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE))
    return img, (small.astype(np.float32) / 255.0 * 2.0 - 1.0)[None]


def _loc_box(locs: list[int], w: int, h: int):
    """PaliGemma <loc> y0 x0 y1 x1 (0..1023 on the padded 224 image) -> pixel box on the original frame."""
    ratio = max(w / IMG_SIZE, h / IMG_SIZE)
    pad_y = (IMG_SIZE - int(h / ratio)) // 2
    pad_x = (IMG_SIZE - int(w / ratio)) // 2
    y0, x0, y1, x1 = (loc / 1024 * IMG_SIZE for loc in locs)
    return [(x0 - pad_x) * ratio, (y0 - pad_y) * ratio, (x1 - pad_x) * ratio, (y1 - pad_y) * ratio]


def run(args: Run) -> None:
    import sentencepiece

    from openpi.shared import download

    tok_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tok_path))
    probe_path = pathlib.Path(args.probe_set)
    probes = json.loads(probe_path.read_text())["frames"]
    out = pathlib.Path(args.out) / args.name
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    model = _load_model(args)
    encode, forward = _make_fns(model)

    def prompt_ids(prefix: str) -> list[int]:
        return [sp.bos_id(), *sp.encode(prefix + "\n")]

    def generate(img_tokens, prefix: str) -> list[int]:
        ids = prompt_ids(prefix)
        n = len(ids)
        for _ in range(args.max_new):
            argmax, _ = forward(img_tokens, *_pack(ids, n))
            nxt = int(argmax[0, len(ids) - 1])
            if nxt == EOS or len(ids) == TEXT_LEN:
                break
            ids.append(nxt)
        return ids[n:]

    def score(img_tokens, prefix: str, answer: str) -> float:
        ids = prompt_ids(prefix)
        n = len(ids)
        ids = ids + sp.encode(answer) + [EOS]
        _, next_logp = forward(img_tokens, *_pack(ids, n))
        return float(np.asarray(next_logp)[0, n - 1 : len(ids) - 1].sum())

    def choose(img_tokens, question: str, options: list[str]) -> tuple[str, dict[str, float]]:
        scores = {o: score(img_tokens, f"answer en {question}", o) for o in options}
        return max(scores, key=scores.get), scores

    rows = []
    for frame in probes:
        img, small = _preprocess(probe_path.parent / frame["image"])
        img_tokens = encode(small)
        h, w = img.shape[:2]
        caption = sp.decode(generate(img_tokens, "caption en"))
        rows.append({"frame": frame["id"], "kind": "caption", "prompt": "caption en", "output": caption})

        overlay = Image.fromarray(img)
        draw = ImageDraw.Draw(overlay)
        for i, obj in enumerate(frame.get("detect", [])):
            gen = generate(img_tokens, f"detect {obj}")
            locs = [t - LOC0 for t in gen if LOC0 <= t < LOC0 + 1024]
            box = _loc_box(locs[:4], w, h) if len(locs) >= 4 else None
            if box:
                color = COLORS[i % len(COLORS)]
                draw.rectangle(box, outline=color, width=3)
                draw.text((box[0] + 3, box[1] + 3), obj, fill=color)
            rows.append(
                {
                    "frame": frame["id"],
                    "kind": "detect",
                    "prompt": f"detect {obj}",
                    "output": sp.decode(gen),
                    "box": box,
                }
            )
        draw.text((8, 8), f"{args.name}: {caption[:90]}", fill="#ffff00")
        overlay.save(out / "overlays" / f"{frame['id']}.png")

        questions = [(f"is there a {o} on the table?", ["yes", "no"], "yes") for o in frame.get("yes", [])]
        questions += [(f"is there a {o} on the table?", ["yes", "no"], "no") for o in frame.get("no", [])]
        questions += [(c["q"], c["options"], c["answer"]) for c in frame.get("choice", [])]
        for q, options, answer in questions:
            pred, scores = choose(img_tokens, q, options)
            kind = "yesno" if options == ["yes", "no"] else "choice"
            rows.append(
                {
                    "frame": frame["id"],
                    "kind": kind,
                    "prompt": q,
                    "expected": answer,
                    "output": pred,
                    "correct": pred == answer,
                    "scores": scores,
                }
            )
        logging.info("%s: %s", frame["id"], caption)

    (out / "results.json").write_text(json.dumps(rows, indent=1))
    for kind in ("yesno", "choice"):
        r = [x for x in rows if x["kind"] == kind]
        if r:
            logging.info("%s accuracy: %d/%d", kind, sum(x["correct"] for x in r), len(r))
    logging.info("Results in %s", out)


def compare(args: Compare) -> None:
    a, b = args.runs
    ra, rb = (json.loads((pathlib.Path(args.out) / n / "results.json").read_text()) for n in (a, b))
    out = pathlib.Path(args.out) / f"compare_{a}_vs_{b}"
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"# Backbone probe: {a} vs {b}", "", f"| | {a} | {b} |", "|---|---|---|"]
    for kind in ("yesno", "choice"):
        ka, kb = ([x for x in r if x["kind"] == kind] for r in (ra, rb))
        if ka:
            lines.append(
                f"| {kind} accuracy | {sum(x['correct'] for x in ka)}/{len(ka)} | {sum(x['correct'] for x in kb)}/{len(kb)} |"
            )
    lines += [
        "",
        "## Answers that differ",
        "",
        f"| frame | kind | prompt | expected | {a} | {b} |",
        "|---|---|---|---|---|---|",
    ]
    for xa, xb in zip(ra, rb, strict=True):
        if xa["kind"] != "detect" and xa["output"] != xb["output"]:
            lines.append(
                f"| {xa['frame']} | {xa['kind']} | {xa['prompt']} | {xa.get('expected', '')} | {xa['output']} | {xb['output']} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    for p in sorted((pathlib.Path(args.out) / a / "overlays").glob("*.png")):
        ia, ib = Image.open(p), Image.open(pathlib.Path(args.out) / b / "overlays" / p.name)
        both = Image.new("RGB", (ia.width + ib.width, max(ia.height, ib.height)))
        both.paste(ia, (0, 0))
        both.paste(ib, (ia.width, 0))
        both.save(out / p.name)
    print("\n".join(lines))
    print(f"\nSide-by-side overlays in {out}")


def main(args: Run | Compare) -> None:
    if isinstance(args, Run):
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Run | Compare))
