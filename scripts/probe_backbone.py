"""Probe what a checkpoint's PaliGemma backbone still knows about our scenes.

Runs the vision encoder + backbone of a pi0/pi0.5 checkpoint as a plain PaliGemma on the frames of a probe set. The
action expert is not used. Per frame and image layout it:
- generates a caption ("caption en"), one box per "detect" object, and free text after each "gen" prompt;
- scores yes/no ("answer en is there a X on the table?") and multiple-choice questions by the log-likelihood of each
  answer (robust to a backbone that no longer writes clean text);
- scores "Subtask: pick up X" for the same present/absent objects with the images visible minus with the images
  masked out. This is the format pi0.5 itself was trained to write, and the difference removes the prior of the words.

Layouts: "single" = the external image only (PaliGemma style); "train" = external + wrist image, the layout the
policy is finetuned on (frames without a wrist image are skipped).

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

TEXT_LEN = 64  # fixed text length (prompt + answer), so each function compiles once per layout
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
    # Image layouts to run: "single" (external only) and/or "train" (external + wrist).
    layouts: tuple[str, ...] = ("single", "train")
    # Prompt before "Subtask: pick up X" in the subtask score.
    subtask_prefix: str = "Task: tidy up the table"
    # Max tokens generated for captions, detections and "gen" prompts.
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
    def forward(state, img_tokens, img_mask, text, text_mask, text_ar):
        llm = nnx.merge(graphdef, state).PaliGemma.llm
        txt_emb = llm(text, method="embed")
        n_img = img_tokens.shape[1]
        emb = jnp.concatenate([img_tokens.astype(txt_emb.dtype), txt_emb], axis=1)
        # Masked image tokens are dropped from attention and positions, as openpi does for missing cameras.
        input_mask = jnp.concatenate([jnp.broadcast_to(img_mask, (1, n_img)), text_mask], axis=1)
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


def _auc(pos, neg) -> float:
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean()) if pos.size and neg.size else float("nan")


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
    visible, hidden = np.ones((1, 1), bool), np.zeros((1, 1), bool)

    def prompt_ids(prefix: str) -> list[int]:
        return [sp.bos_id(), *sp.encode(prefix + "\n")]

    def generate(img_tokens, prefix: str) -> list[int]:
        ids = prompt_ids(prefix)
        n = len(ids)
        for _ in range(args.max_new):
            argmax, _ = forward(img_tokens, visible, *_pack(ids, n))
            nxt = int(argmax[0, len(ids) - 1])
            if nxt == EOS or len(ids) == TEXT_LEN:
                break
            ids.append(nxt)
        return ids[n:]

    def score(img_tokens, prefix: str, answer: str, img_mask=visible) -> float:
        ids = prompt_ids(prefix)
        n = len(ids)
        ids = ids + sp.encode(answer) + [EOS]
        _, next_logp = forward(img_tokens, img_mask, *_pack(ids, n))
        return float(np.asarray(next_logp)[0, n - 1 : len(ids) - 1].sum())

    rows = []
    for layout in args.layouts:
        for frame in probes:
            if layout == "train" and "wrist" not in frame:
                continue
            img, small = _preprocess(probe_path.parent / frame["image"])
            img_tokens = encode(small)
            if layout == "train":
                img_tokens = np.concatenate([img_tokens, encode(_preprocess(probe_path.parent / frame["wrist"])[1])], 1)
            base = {"frame": frame["id"], "layout": layout}
            h, w = img.shape[:2]
            caption = sp.decode(generate(img_tokens, "caption en"))
            rows.append(base | {"kind": "caption", "prompt": "caption en", "output": caption})
            rows.extend(
                base | {"kind": "gen", "prompt": p, "output": sp.decode(generate(img_tokens, p))}
                for p in frame.get("gen", [])
            )

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
                rows.append(base | {"kind": "detect", "prompt": f"detect {obj}", "output": sp.decode(gen), "box": box})
            draw.text((8, 8), f"{args.name} [{layout}]: {caption[:90]}", fill="#ffff00")
            overlay.save(out / "overlays" / f"{frame['id']}_{layout}.png")

            objects = [(o, "yes") for o in frame.get("yes", [])] + [(o, "no") for o in frame.get("no", [])]
            questions = [(f"is there a {o} on the table?", ["yes", "no"], e) for o, e in objects]
            questions += [(c["q"], c["options"], c["answer"]) for c in frame.get("choice", [])]
            for q, options, answer in questions:
                scores = {o: score(img_tokens, f"answer en {q}", o) for o in options}
                pred = max(scores, key=scores.get)
                kind = "yesno" if options == ["yes", "no"] else "choice"
                rows.append(
                    base
                    | {
                        "kind": kind,
                        "prompt": q,
                        "expected": answer,
                        "output": pred,
                        "correct": pred == answer,
                        "scores": scores,
                    }
                )
            for o, e in objects:
                answer = f"Subtask: pick up {o}"
                scores = {
                    m: score(img_tokens, args.subtask_prefix, answer, mask)
                    for m, mask in (("image", visible), ("blind", hidden))
                }
                rows.append(
                    base
                    | {
                        "kind": "subtask",
                        "prompt": o,
                        "expected": e,
                        "output": scores["image"] - scores["blind"],
                        "scores": scores,
                    }
                )
            logging.info("%s [%s]: %s", frame["id"], layout, caption)

    (out / "results.json").write_text(json.dumps(rows, indent=1))
    for layout in args.layouts:
        lr = [x for x in rows if x["layout"] == layout]
        for kind in ("yesno", "choice"):
            r = [x for x in lr if x["kind"] == kind]
            if r:
                logging.info("[%s] %s accuracy: %d/%d", layout, kind, sum(x["correct"] for x in r), len(r))
        st = [x for x in lr if x["kind"] == "subtask"]
        if st:
            auc = _auc(
                [x["output"] for x in st if x["expected"] == "yes"], [x["output"] for x in st if x["expected"] == "no"]
            )
            logging.info("[%s] subtask image-vs-blind AUC present vs absent: %.2f", layout, auc)
    logging.info("Results in %s", out)


def compare(args: Compare) -> None:
    a, b = args.runs
    ra, rb = (json.loads((pathlib.Path(args.out) / n / "results.json").read_text()) for n in (a, b))
    out = pathlib.Path(args.out) / f"compare_{a}_vs_{b}"
    out.mkdir(parents=True, exist_ok=True)

    def auc(rows):
        return _auc(
            [x["output"] for x in rows if x["expected"] == "yes"], [x["output"] for x in rows if x["expected"] == "no"]
        )

    lines = [f"# Backbone probe: {a} vs {b}", "", f"| layout | measure | {a} | {b} |", "|---|---|---|---|"]
    for layout in sorted({x.get("layout", "single") for x in ra}):
        la, lb = ([x for x in r if x.get("layout", "single") == layout] for r in (ra, rb))
        for kind in ("yesno", "choice"):
            ka, kb = ([x for x in r if x["kind"] == kind] for r in (la, lb))
            if ka:
                lines.append(
                    f"| {layout} | {kind} accuracy | {sum(x['correct'] for x in ka)}/{len(ka)} | "
                    f"{sum(x['correct'] for x in kb)}/{len(kb)} |"
                )
        sa, sb = ([x for x in r if x["kind"] == "subtask"] for r in (la, lb))
        if sa:
            lines.append(f"| {layout} | subtask AUC (image vs blind) | {auc(sa):.2f} | {auc(sb):.2f} |")
    lines += [
        "",
        "## Answers that differ",
        "",
        f"| layout | frame | kind | prompt | expected | {a} | {b} |",
        "|---|---|---|---|---|---|---|",
    ]
    for xa, xb in zip(ra, rb, strict=True):
        if xa["kind"] not in ("detect", "subtask") and xa["output"] != xb["output"]:
            lines.append(
                f"| {xa.get('layout', 'single')} | {xa['frame']} | {xa['kind']} | {xa['prompt']} | "
                f"{xa.get('expected', '')} | {xa['output']} | {xb['output']} |"
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
