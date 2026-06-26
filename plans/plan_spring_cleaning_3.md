## Questions about main.py
- Line 19, artifact stem business. I want this to follow the nomenclature standard we defined in @plans/completed/plan_spring_cleaning.md. I want to be _certain_ that there will not be naming collisions if I try

{{**What `artifact_stem` actually is, and where it can collide.** It's a
*readable label string* built once per run (`build_artifact_stem`, main.py:19-32)
and stored at `config['artifact_stem']`. Every use:
- `create_diagnostics.py:81` → seeded into diagnostics `static_context`.
- `methods/diagnostics/ntk.py:124` → the **only** place it becomes a filename:
  `spectrum/{dataset_name}/{artifact_stem}.p`.

Per §2 of the completed plan, **uniqueness comes from the run dir**
(`experiments/run_<timestamp>[_<n>]`, claimed by an atomic
`os.makedirs(exist_ok=False)`). Anything written *under* the run dir is
collision-free **by construction** — two runs with byte-identical hyperparameters
still land in different run dirs. So under `run_dir/`, the stem need not be
globally unique; it's just a human-readable tag.}}

{{**The one real hazard.** `ntk.py:116-124` writes the spectrum **outside** the
run dir, into a shared tree at `{project_root}/spectrum/{dataset}/{stem}.p`, with
`os.makedirs(exist_ok=True)` + a plain write — **no §2 write-guard**. So if two
runs share all stem fields (bsel/seed/model/opt/bs/ratio/lr/wd/noise_percent)
*and* both enable `save_spectrum`, the second **silently overwrites** the first —
violating "collisions are bugs, and bugs are loud." It's latent today because
`save_spectrum` defaults to `False` (ntk.py:50), but it's exactly the collision
you're worried about. (`ntk_indices/{dataset}.npy`, ntk.py:111, is a deliberate
per-dataset *cache* — shared on purpose, fine, though also unguarded.)}} [[Change this!! These should save in the run dir, not outside, and just be called `<run_dir>/spectrum.p` if saved at all]]

{{**To be *certain* of no collisions, pick one (I recommend A):**
- **A — put the spectrum under the run dir** (e.g.
  `run_dir/snapshots/spectrum_<stem>.p` or `run_dir/spectrum/`). The atomic run
  dir then guarantees uniqueness; the stem becomes a pure in-dir label and its
  format is purely cosmetic. This is the plan-consistent fix (§6: all run outputs
  live under the run dir).
- **B — keep `spectrum/` shared but make it loud + unique:** route the write
  through `run_dir.write_guard` (so an existing target raises) and fold the run
  id/seed into the stem. More fragile than A.}}

{{**On the nomenclature *format*.** Today the stem is **JSON**
(`json.dumps({...}).replace(' ','')` → `{"bsel":"Uniform","seed":42,...}`), which
is *not* the §8 standard. §8 defines the filename grammar as
`<k1><v1>_<k2><v2>...` (underscore-joined key-value fragments, dotted keys
sanitized), e.g. `bselUniform_seed42_modelLinear_optSGD_bs320_...`. Converting
`build_artifact_stem` to emit that is small. **Decide first:**
`scripts/backfill_alpha_wandb.py` *parses the stem as JSON* (`json.loads`) to
recover `alpha`; switching grammar breaks it unless we (i) also store those fields
in `config`/`wandb.config` (cleaner — nothing round-trips through the filename),
or (ii) update that parser. I'd fold the determining fields into `config`
explicitly and let the stem be **display-only**.}}

{{**Dead flag:** `--artifact_suffix` (main.py:143-144) is parsed but **never
consumed** — `build_artifact_stem(config)` ignores `args`. Its help claims it
merges extra fields into the stem. I'd delete it unless you want it wired in.}} [[Delete it]]

{{**Net:** Not certain as-is — the shared `spectrum/` write is an unguarded
overwrite path. Adopt **A** and the run dir's atomic claim makes collisions
*impossible* regardless of stem format; then reformat the stem to §8 `<k><v>` as
a readable label. Want me to spec concrete plan items (A + stem reformat + drop
`--artifact_suffix`)? [[ ]]}}
- Line 121. If I only use 2 gpus on a cluster with many more available, will this set the seed on all the gpus? I mean if I set CUDA_VISIBLE_DEVICES=[0, 2], will it set the seed on gpu 1 as well (not desired)?

{{**No — it cannot touch GPU 1.** `CUDA_VISIBLE_DEVICES` masks at the driver
level: a process started with `CUDA_VISIBLE_DEVICES=0,2` sees **exactly two**
devices, remapped to logical `cuda:0`→physical 0 and `cuda:1`→physical 2.
Physical GPU 1 is not enumerable and not addressable — it doesn't exist as far as
this process is concerned.

So in `init_seeds`:
- `torch.cuda.device_count()` returns **2** (the visible count), so the `> 1`
  branch runs.
- `torch.cuda.manual_seed_all(seed)` seeds **all visible** devices — the two you
  asked for (physical 0 and 2). It iterates logical indices `0..count-1`, which
  never includes the masked-out physical GPU 1.

`torch.cuda.manual_seed(seed)` (line 120) seeds only the *current* visible device
(`cuda:0`); `manual_seed_all` then covers both visible ones — which is what you
want for reproducibility across the GPUs you're actually using. Nothing here ever
reaches a masked GPU. (Minor: the env-var syntax is comma-separated
`CUDA_VISIBLE_DEVICES=0,2`, not a Python list `[0,2]`.) **No change needed.**}}