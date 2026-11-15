"""Probe how a finetuned policy uses its prompt, and what its backbone K/V still encodes about objects.

Works on training frames, passed through the exact training transforms. Three steps:

    # 1. Laptop (CPU, has the dataset): pick 2 frames per episode (one while reaching, one while carrying), save them
    #    after the training data transforms, and write the prompt variants for each frame.
    uv run scripts/probe_policy.py extract --root ~/datasets/FilippoGorini/vla_kinova_gen3_clutter_pick_place@v2.1 \
        --out ../../analysis/backbone_probe/policy

    # 2. GPU, one checkpoint per process:
    #    loss:     flow-matching loss of the recorded actions under each prompt variant, with the same noise and
    #              flow time for every variant and checkpoint (paired), so loss(variant) - loss(original) says how
    #              much the actions depend on the edited words;
    #    features: V of every backbone layer (what the action expert attends to) and the SigLIP output, mean-pooled
    #              per camera, on the carrying frames with an empty task and a fixed state (no text information).
    uv run scripts/probe_policy.py loss --checkpoint hf://FilippoGorini/pi05_kinova_clutter_rtc/30000 --name 30k \
        --samples ../../analysis/backbone_probe/policy
    uv run scripts/probe_policy.py features --checkpoint hf://FilippoGorini/pi05_kinova_clutter_rtc/30000 --name 30k \
        --samples ../../analysis/backbone_probe/policy

    # 3. Laptop: analysis/backbone_probe/analyze_policy.py (loss table, linear probe of the held object per layer).

pi05_base was never trained on this robot, so its loss numbers are not comparable: use it for `features` only.
"""

import dataclasses
import json
import logging
import pathlib
import random
import re

import numpy as np
import tyro

COLOURS = ["red", "green", "blue", "black", "yellow", "pink", "purple", "white", "orange", "grey", "brown",
           "light-blue", "clear", "transparent"]  # fmt: skip
_COL = "(?:" + "|".join(COLOURS) + ")"
# Words that make an object description depend on the scene (position, size, function): not swapped.
RELATIONAL = {"leftmost", "rightmost", "closest", "smaller", "smallest", "bigger", "big", "small", "next", "behind",
              "under", "between", "something", "object", "tool", "to", "with"}  # fmt: skip
PICK = re.compile(
    r"^(?P<verb>pick up|pick|grab|take|get) (?P<art>(?:the|a) )?(?P<obj>.+?) and (?P<v2>put|place|set) (?:it|them) "
    r"(?P<where>(?P<prep>in|on top of|on|over) the (?P<dest>.+))$"
)
PUT = re.compile(
    r"^(?P<verb>put|place) (?P<art>the )(?P<obj>.+?) (?P<where>(?P<prep>in|on top of|on|over) the (?P<dest>.+))$"
)
OBJ = re.compile(rf"^(?:(?P<colour>{_COL}(?: and {_COL})?) )?(?P<noun>[a-z\- ]+)$")
OTHER_VERB = {"pick up": "grab", "pick": "grab", "grab": "take", "take": "pick", "get": "pick", "put": "place",
              "place": "put"}  # fmt: skip


def parse(prompt: str) -> dict | None:
    """Split a task into its parts (with spans); colour and noun only for plain object descriptions."""
    m = PICK.match(prompt) or PUT.match(prompt)
    if m is None:
        return None
    p = {k: m[k] for k in ("verb", "art", "obj", "where", "prep", "dest")} | {"colour": None, "noun": None}
    p["art"] = p["art"] or ""
    p["span"] = {k: m.span(k) for k in ("verb", "obj", "where")}
    o = OBJ.match(m["obj"])
    if o and not RELATIONAL & set(o["noun"].split()) and len(o["noun"].split()) <= 3:
        p |= {"colour": o["colour"], "noun": o["noun"]}
    return p


def variants(prompt: str, vocab: dict, rng: random.Random) -> dict[str, str]:
    """Edited prompts, each changing one part of the original string. Left out when it does not apply."""
    out = {"orig": prompt, "empty": ""}
    p = parse(prompt)
    if p is None:
        return out

    def edit(part: str, new: str) -> str:
        a, b = p["span"][part]
        return prompt[:a] + new + prompt[b:]

    out["verb_swap"] = edit("verb", OTHER_VERB[p["verb"]])  # wording seen in training, same content
    out["new_template"] = f"move {p['art']}{p['obj']} to the {p['dest']}"  # unseen wording, same content
    out["pick_only"] = f"pick up {p['art']}{p['obj']}"  # pi0.5 subtask style, no destination
    out["dest_swap"] = edit("where", rng.choice([w for w in vocab["where"] if w != p["where"]]))
    if p["noun"]:
        colour = f"{p['colour']} " if p["colour"] else ""
        out["noun_swap"] = edit("obj", colour + rng.choice([n for n in vocab["noun"] if n != p["noun"]]))
        if p["colour"]:
            other = rng.choice([c for c in COLOURS[:11] if c not in p["colour"].split()])
            out["colour_swap"] = edit("obj", f"{other} {p['noun']}")
            out["colour_drop"] = edit("obj", p["noun"])
            out["noun_drop"] = edit("obj", f"{p['colour']} object")
    return out


def make_vocab(prompts) -> dict:
    """Nouns of plain object descriptions, and destinations of pick-and-place tasks (not stacking ones)."""
    parsed = [(t, p) for t in prompts if (p := parse(t))]
    return {
        "noun": sorted({p["noun"] for _, p in parsed if p["noun"]}),
        "where": sorted({p["where"] for t, p in parsed if PICK.match(t) and p["prep"] in ("in", "on")}),
    }


@dataclasses.dataclass
class Extract:
    """Pick frames from the training dataset and save them after the training data transforms (laptop, CPU)."""

    # Local LeRobot v2.1 dataset folder (the one the checkpoint was trained on).
    root: str
    out: str = "probe_policy"
    config: str = "pi05_kinova_clutter_rtc"
    seed: int = 0
    # Gripper command above this = closed (commands are 0.0 / 0.8).
    closed: float = 0.4
    # Only print the variants of every task in the dataset, extract nothing.
    prompts_only: bool = False


@dataclasses.dataclass
class Loss:
    """Paired flow-matching loss of the recorded actions under each prompt variant (GPU)."""

    # Checkpoint (local path, gs://, hf://), or "dummy" for a tiny random model (code-path test).
    checkpoint: str
    name: str
    samples: str = "probe_policy"
    config: str = "pi05_kinova_clutter_rtc"
    # Noise / flow-time draws per sample (shared by every variant and checkpoint).
    draws: int = 8
    batch: int = 32
    seed: int = 0


@dataclasses.dataclass
class Features:
    """Per-layer backbone V and SigLIP output, mean-pooled per camera, on the carrying frames (GPU)."""

    checkpoint: str
    name: str
    samples: str = "probe_policy"
    config: str = "pi05_kinova_clutter_rtc"
    batch: int = 16


def extract(args: Extract) -> None:
    from lerobot.common.datasets import lerobot_dataset

    from openpi import transforms as _transforms
    from openpi.training import config as _config

    root = pathlib.Path(args.root).expanduser()
    tasks = [json.loads(line)["task"] for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()]
    vocab = make_vocab(tasks)
    if args.prompts_only:
        rng = random.Random(args.seed)
        for t in tasks:
            print(json.dumps(variants(t, vocab, rng)))
        print(json.dumps(vocab, indent=1))
        return

    cfg = _config.get_config(args.config)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    meta = lerobot_dataset.LeRobotDatasetMetadata(dc.repo_id, root=root)
    fetch = cfg.model.action_horizon + max(0, dc.action_state_delay_fetch_extra)
    ds = lerobot_dataset.LeRobotDataset(
        dc.repo_id,
        root=root,
        delta_timestamps={k: [t / meta.fps for t in range(fetch)] for k in dc.action_sequence_keys},
    )
    # Same transforms as training up to (not including) tokenization, which `loss` redoes per prompt variant.
    transform = _transforms.compose(
        [
            _transforms.PromptFromLeRobotTask(meta.tasks),
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            _transforms.ResizeImages(224, 224),
        ]
    )

    actions = np.asarray(ds.hf_dataset.with_format("numpy")["action"])
    episodes = np.asarray(ds.hf_dataset.with_format("numpy")["episode_index"])
    rng, np_seed = random.Random(args.seed), np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # ShiftActions draws its delay from the global numpy RNG
    picks = []
    for ep in np.unique(episodes):
        idx = np.flatnonzero(episodes == ep)
        closed = actions[idx, -1] > args.closed
        edges = np.flatnonzero(np.diff(closed.astype(int)))  # 0->1 and 1->0 switches
        starts, ends = [e + 1 for e in edges if not closed[e]], [e + 1 for e in edges if closed[e]]
        if not starts:
            continue
        g0 = starts[0]
        picks.append((int(idx[int(np_seed.uniform(0.2, 0.8) * g0)]), "reach"))
        # Carry = the longest closed interval (skips failed grasps before a retry), away from its edges.
        a, b = max(((s, next((e for e in ends if e > s), len(idx))) for s in starts), key=lambda x: x[1] - x[0])
        if b - a > 40:
            picks.append((int(idx[rng.randint(a + 15, b - 15)]), "carry"))

    rows, arrays = [], {}
    for i, (index, phase) in enumerate(picks):
        x = transform(ds[index])
        for k in ("state", "actions"):
            arrays.setdefault(k, []).append(np.asarray(x[k], np.float32))
        for cam, img in x["image"].items():
            arrays.setdefault(f"image/{cam}", []).append(np.asarray(img))
            arrays.setdefault(f"image_mask/{cam}", []).append(np.asarray(x["image_mask"][cam]))
        prompt = str(x["prompt"])
        rows.append(
            {"i": i, "index": index, "episode": int(episodes[index]), "phase": phase, "prompt": prompt}
            | {"parse": parse(prompt), "variants": variants(prompt, vocab, rng)}
        )
        if i % 100 == 0:
            logging.info("%d/%d %s %s", i, len(picks), phase, prompt)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", **{k: np.stack(v) for k, v in arrays.items()})
    (out / "samples.json").write_text(json.dumps({"vocab": vocab, "samples": rows}, indent=1))
    logging.info("%d samples (%d carry) in %s", len(rows), sum(r["phase"] == "carry" for r in rows), out)


def _load(checkpoint: str, config: str):
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.shared import download
    from openpi.training import config as _config

    cfg = _config.get_config(config)
    if checkpoint == "dummy":  # tiny random model, code-path test only
        import jax

        mc = dataclasses.replace(cfg.model, paligemma_variant="dummy", action_expert_variant="dummy")
        return dataclasses.replace(cfg, model=mc), mc.create(jax.random.key(0))
    path = download.maybe_download(checkpoint, ignore_patterns=("train_state/*",))
    params_dir = path if path.name == "params" else path / "params"
    model = cfg.model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))
    return cfg, model


def _batches(args, rows, data, cfg, prompt_of):
    """Model inputs per batch of rows, through the training model transforms. prompt_of(row) -> (prompt, state or
    None for the recorded state)."""
    from openpi import transforms as _transforms
    from openpi.models import model as _model

    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    tf = _transforms.compose(dc.model_transforms.inputs)
    cams = sorted({k.split("/")[1] for k in data.files if k.startswith("image/")})
    for s in range(0, len(rows), args.batch):
        chunk = rows[s : s + args.batch]
        chunk = chunk + [chunk[-1]] * (args.batch - len(chunk))  # fixed batch: one compile
        items = []
        for r in chunk:
            i, (prompt, state) = r["i"], prompt_of(r)
            items.append(
                tf(
                    {
                        "image": {c: data[f"image/{c}"][i] for c in cams},
                        "image_mask": {c: data[f"image_mask/{c}"][i] for c in cams},
                        "state": data["state"][i] if state is None else state,
                        "actions": data["actions"][i],
                        "prompt": prompt,
                    }
                )
            )
        batch = {
            k: np.stack([x[k] for x in items])
            for k in ("state", "actions", "tokenized_prompt", "tokenized_prompt_mask")
        }
        batch["image"] = {c: np.stack([x["image"][c] for x in items]) for c in cams}
        batch["image_mask"] = {c: np.stack([x["image_mask"][c] for x in items]) for c in cams}
        yield len(rows[s : s + args.batch]), _model.Observation.from_dict(batch), batch["actions"]


def loss(args: Loss) -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    sdir = pathlib.Path(args.samples)
    rows = json.loads((sdir / "samples.json").read_text())["samples"]
    data = np.load(sdir / "samples.npz")
    cfg, model = _load(args.checkpoint, args.config)
    graphdef, state = nnx.split(model)
    n_dims = data["actions"].shape[-1]  # real action dims; the rest is padding

    @jax.jit
    def per_sample(state, obs, actions, noise, time):
        """Pi0RTC.compute_loss without the RTC prefix (delay 0), per sample and draw, on the real action dims.

        The prefix (images + prompt) never attends to the suffix, so its KV cache is computed once and shared by all
        draws, as in sample_actions. noise (b, draws, ah, ad), time (b, draws) -> loss (b, draws)."""
        m = nnx.merge(graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)
        pre, pre_mask, pre_ar = m.embed_prefix(obs)
        _, kv = m.PaliGemma.llm(
            [pre, None], mask=make_attn_mask(pre_mask, pre_ar), positions=jnp.cumsum(pre_mask, 1) - 1
        )

        def draw(nt):
            noise, time = nt
            t = jnp.broadcast_to(time[:, None], actions.shape[:2])
            x_t = t[..., None] * noise + (1 - t[..., None]) * actions
            suf, suf_mask, suf_ar, cond = m.embed_suffix(obs, x_t, t)
            s = suf.shape[1]
            mask = jnp.concatenate(
                [
                    jnp.broadcast_to(pre_mask[:, None], (len(pre_mask), s, pre_mask.shape[1])),
                    make_attn_mask(suf_mask, suf_ar),
                ],
                axis=-1,
            )
            positions = jnp.sum(pre_mask, axis=-1)[:, None] + jnp.cumsum(suf_mask, axis=-1) - 1
            (_, out), _ = m.PaliGemma.llm(
                [None, suf], mask=mask, positions=positions, kv_cache=kv, adarms_cond=[None, cond]
            )
            v_t = m.action_out_proj(out[:, -m.action_horizon :])
            return jnp.mean(jnp.square(v_t - (noise - actions))[..., :n_dims], axis=(1, 2))

        return jax.lax.map(draw, (noise.swapaxes(0, 1), time.T)).T

    # One (noise, time) set per sample and draw, from the sample index: identical for every variant and checkpoint.
    ah, ad = cfg.model.action_horizon, cfg.model.action_dim
    keys = [jax.random.fold_in(jax.random.key(args.seed), r["i"]) for r in rows]
    noise = np.stack([np.asarray(jax.random.normal(k, (args.draws, ah, ad))) for k in keys])
    time = np.stack(
        [np.asarray(jax.random.beta(jax.random.fold_in(k, 1), 1.5, 1, (args.draws,)) * 0.999 + 0.001) for k in keys]
    )

    names = sorted({v for r in rows for v in r["variants"]})
    result = {r["i"]: {} for r in rows}
    for name in names:
        sel = [r for r in rows if name in r["variants"]]
        done = 0
        for n, obs, actions in _batches(args, sel, data, cfg, lambda r, name=name: (r["variants"][name], None)):
            ids = [r["i"] for r in sel[done : done + n]]
            ids += [ids[-1]] * (args.batch - n)
            losses = np.asarray(per_sample(state, obs, jnp.asarray(actions), noise[ids], time[ids]))
            for j, i in enumerate(ids[:n]):
                result[i][name] = float(losses[j].mean())
            done += n
        logging.info("%s: %d samples, mean loss %.4f", name, len(sel), np.mean([result[r["i"]][name] for r in sel]))

    out = sdir / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_loss.json").write_text(json.dumps({"checkpoint": args.checkpoint, "loss": result}, indent=1))
    logging.info("Saved %s", out / "prompt_loss.json")


def features(args: Features) -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    sdir = pathlib.Path(args.samples)
    rows = [r for r in json.loads((sdir / "samples.json").read_text())["samples"] if r["phase"] == "carry"]
    data = np.load(sdir / "samples.npz")
    cfg, model = _load(args.checkpoint, args.config)
    graphdef, state = nnx.split(model)

    @jax.jit
    def run(state, obs):
        m = nnx.merge(graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)
        pre, pre_mask, pre_ar = m.embed_prefix(obs)
        _, (_, v) = m.PaliGemma.llm(
            [pre, None], mask=make_attn_mask(pre_mask, pre_ar), positions=jnp.cumsum(pre_mask, 1) - 1
        )
        # v: (layers, b, seq, kv_heads, head_dim). Image tokens come first, 256 per camera in obs.images order.
        n = 256
        v_pool = jnp.stack([v[:, :, c * n : (c + 1) * n].mean(axis=2)[..., 0, :] for c in range(len(obs.images))], 2)
        sig = jnp.stack([m.PaliGemma.img(img, train=False)[0].mean(axis=1) for img in obs.images.values()], 1)
        return v_pool.astype(jnp.float32), sig.astype(jnp.float32)

    # Empty task and a fixed (all-zero, i.e. mid-range) state: the pooled features carry image information only.
    zero_state = np.zeros_like(data["state"][0])
    vs, sigs = [], []
    for n, obs, _ in _batches(args, rows, data, cfg, lambda r: ("", zero_state)):
        v, sig = run(state, obs)
        vs.append(np.asarray(v)[:, :n].transpose(1, 0, 2, 3))  # (b, layers, cams, head_dim)
        sigs.append(np.asarray(sig)[:n])
    out = sdir / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "features.npz",
        v=np.concatenate(vs).astype(np.float16),
        siglip=np.concatenate(sigs).astype(np.float16),
        i=np.asarray([r["i"] for r in rows]),
    )
    logging.info("Saved %s: %d carry frames", out / "features.npz", len(rows))


def main(args: Extract | Loss | Features) -> None:
    {Extract: extract, Loss: loss, Features: features}[type(args)](args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Extract | Loss | Features))
