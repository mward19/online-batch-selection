We would like to update the nomenclature of a lot of things.

The config should have this key added, which if not provided raises an error.
```yaml
run_name_format:
    - "run"
    - dataset.name
    - method
    - training_opt.optimizer
    - "lr":
        - training_opt.optim_params.lr
    - loss_type

```

For configs/clifar3_rholoss.yaml, this should result in the run name "run_CIFAR3_Rholoss_AdamW_lr-0.001_CrossEntropy". Elements separated by "_", key-value pairs separated by "-". (Make these separators parameters in the function you write that does this, so we can change them later if we want.) When a grid search happens with generate_configs.py, this should use the generated configs, not the template. [[Is `"run"` different than `run` in YAML? How else can I differentiate between string literals and config parameters?]]

{{**No — `"run"` and `run` are the *same string* in YAML.** Quoting a bareword
doesn't change its type; both parse to the Python string `"run"`. (Quotes only
matter when the unquoted form would parse as something else — `"123"`→str vs
`123`→int, `"yes"`/`"null"`→str vs bool/null. `run` is already a string, so quotes
are a no-op.) So we **cannot** use quoting to tell a literal from a config path.

We need an explicit rule. Two viable schemes:

- **Scheme A — sigil-marks-a-config-reference (my recommendation).** A token is a
  config path *iff* it starts with a marker (say `$`); everything else is a
  literal. Unambiguous, and a typo'd path (`$dataset.naem`) raises loudly instead
  of silently becoming part of the name:
  ```yaml
  run_name_format:
    - run                               # literal
    - $dataset.name                     # config ref
    - $method
    - $training_opt.optimizer
    - lr: [$training_opt.optim_params.lr]   # key-value pair
    - $training_opt.loss_type
  ```
- **Scheme B — resolve-or-literal heuristic (keeps your exact syntax).** A token
  is a config ref if it resolves against the config, else a literal — but with the
  guard that **any token containing a `.` MUST resolve or we raise** (a dotted
  token is unambiguously path-intended, so a miss is a typo, not a literal). Bare
  tokens (`method`, `run`) resolve-or-literal. This reproduces your example
  exactly, but a *bare* typo (`methdo`) silently becomes a literal.

I lean **A** (loud on every typo; self-documenting). Which do you want? [[A! love it]]}}

{{**Two corrections to the example, regardless of scheme:**
1. `loss_type` resolves to `training_opt.loss_type` in `cifar3_rholoss.yaml` (it's
   nested, not top-level), so the path must be `training_opt.loss_type`. `method`
   *is* top-level, so bare `method`/`$method` is correct.
2. The config has `method: RhoLoss` (capital L), so the rendered name will be
   `...RhoLoss...`, not `Rholoss`. Values render **verbatim** from the config
   unless you want a lowercasing/normalization option. OK to render verbatim? [[That was just a mistake on my part, raise an exception if it's wrong.]]}}

The run directory names should use this. Using the above example, the run directory would be called "20260626_110000_CIFAR3_Rholoss_AdamW_lr-0.001_CrossEntropy/". Add a flag to main.py that is the base experiments directory (defaults to `./experiments`), but can be changed. If a run name collides with a run in the experiments directory given, add `_1`, `_2`, etc. after the timestamp, i.e., "20260626_110000_1_CIFAR3_Rholoss_AdamW_lr-0.001_CrossEntropy/".

The W&B run names should just be based on this run_name_format (in other words, they might not be unique, and that's okay). Also read my notes in plan_spring_cleaning_3.md about spectrum naming. 

{{**Directory-vs-W&B naming — confirming the rule.** Your two examples:
- W&B run name: `run_CIFAR3_Rholoss_AdamW_lr-0.001_CrossEntropy` (has `run`)
- Run dir:     `20260626_110000_CIFAR3_Rholoss_AdamW_lr-0.001_CrossEntropy` (no `run`,
  timestamp instead), and on collision `20260626_110000_1_CIFAR3_...`.

The consistent reading: the leading literal `run` is a **run-id slot**. In W&B it
renders as the word `run`; on the filesystem the *timestamp* is the run id, so it
takes that slot. So I'll implement:
- `run_name = render(format)` → the W&B name (verbatim, may be non-unique — fine).
- `dir_name = <timestamp>[_<n>] + elem_sep + render(format without a leading
  literal "run")`; the `_<n>` collision suffix sits **right after the timestamp**.

Confirm this "leading `run` → timestamp on the filesystem" rule? (Alternative: keep
`run` in the dir too → `20260626_110000_run_CIFAR3_...`. Your example says drop it.)
[[That was a mistake. If "run" is in the config, include it everywhere.]]}}

{{**Folding in the plan_3 decisions** (you marked them up):
- **Spectrum** → write to `<run_dir>/spectrum.p` (inside the run dir), if saved at
  all. Drop the shared `{project_root}/spectrum/{dataset}/{stem}.p` path entirely.
- **`--artifact_suffix`** → delete the flag.
- Consequence: with the spectrum inside the run dir, `artifact_stem` loses its only
  filename use. It's then vestigial (only seeded into diagnostics context + parsed
  by `scripts/backfill_alpha_wandb.py`). **Proposal:** replace `artifact_stem` with
  the new `run_name` (store `config['run_name']`), drop `build_artifact_stem`, and
  have `backfill_alpha_wandb.py` read `alpha` from `config` instead of parsing a
  stem. Confirm dropping `artifact_stem`? [[Drop it]]}}

What about the `save_labels.py`? What happened to that file?

{{**It was deleted in Phase 2 of the original spring-cleaning refactor** (commit
`2224b93`, "Phase 2: noise-label cache with readable names under ./cache/"). It's
gone on purpose, not lost. Its job — precomputing train/val labels — was replaced
by a **lazy, content-addressed noise cache**: `generate_noise.py` now derives a
readable cache filename from `LABEL_CACHE_KEYS` (dataset, noise_percent,
noise_seed, noise_algo) under `./cache/labels/`, computes-if-absent at data-load
time, validates on read, and symlinks the cache file into each run dir as
`labels`. So there's no separate "save labels" step anymore; caching happens
on demand the first time a dataset/noise combo is loaded. (The old CLAUDE.md
`save_labels.py` command line is stale and should be removed — flagging for a
docs pass.)}} [[Great.]]

---

## Design summary (FINALIZED — decisions baked in; ready to implement on your go)

**Resolved decisions:** Scheme A (`$` sigil for config refs) · values render
**verbatim**, any unresolvable `$path` **raises** · the literal `run` is kept
**everywhere** (W&B name *and* dir) — no stripping · `build_run_name` lives in
`run_dir.py` · `artifact_stem` is **dropped**, replaced by `run_name`.

**1. `run_name_format` (required config key).** Absent → raise at startup. A list
whose elements are: a **literal** string, a **config reference** `$dotted.path`
(Scheme A), or a **key-value dict** `{label: [$refs...]}`.

**2. `build_run_name(config, fmt, *, value_sep="_", kv_sep="-")`** in `run_dir.py`:
- Two separators, both parameters:
  - `value_sep="_"` — joins top-level elements, *and* joins multiple values
    inside one kv-pair.
  - `kv_sep="-"` — joins a kv-pair's label to its value block.
- An element starting with `$` is a config ref: strip `$`, resolve the dotted path
  (reuse `generate_configs._get_dotted`); an unresolvable path **raises**
  (`KeyError`/`ValueError`). A non-`$` string is a literal.
- A kv dict `{label: [$a, $b]}` → `f"{label}{kv_sep}{value_sep.join(values)}"`
  (e.g. `lr-0.001`).
- Join all elements with `value_sep`. Returns the run name (e.g.
  `run_CIFAR3_RhoLoss_AdamW_lr-0.001_CrossEntropy`).

**3. Run directory** (`run_dir.setup_run_dir`):
- New `run_name` param threaded in from `main.py`.
- Dir = **`<timestamp>[_<n>]_<run_name>`** — `run` is kept, timestamp is prefixed:
  `20260626_110000_run_CIFAR3_RhoLoss_AdamW_lr-0.001_CrossEntropy`, collision →
  `20260626_110000_1_run_CIFAR3_...`.
- `_claim_dir` reworked so the `_<n>` suffix lands **right after the timestamp**,
  not at the end: base `f"{timestamp}_{run_name}"`; on collision
  `f"{timestamp}_{n}_{run_name}"`. Still an atomic `makedirs(exist_ok=False)` retry
  loop (§2 race-safety preserved).

**4. Base experiments dir flag** (`main.py`): `--experiments_dir` (default
`./experiments`), passed to `setup_run_dir(experiments_root=...)`. (The function
already takes `experiments_root`; just wire the flag.)

**5. W&B run name** (`main.py`): replace the hardcoded `run.name = f"{method}_..."`
block (lines 219-230) with `run.name = run_name` (the full rendered name, `run`
included). Non-unique is acceptable.

**6. generate_configs.py:** no special-casing — `run_name_format` is a normal
(non-`__REQUIRED__`) key, copied verbatim into every generated config; `main.py`
renders it at runtime against each generated config's *actual* values. Satisfies
"use the generated configs, not the template."

**7. Spectrum + cleanup** (per plan_3 decisions): spectrum → `<run_dir>/spectrum.p`;
delete `--artifact_suffix`; drop `build_artifact_stem`/`config['artifact_stem']`,
replace with `config['run_name']`; update `scripts/backfill_alpha_wandb.py` to read
`alpha` from `config` instead of parsing a stem.

## Checklist
- ~~`build_run_name(config, fmt, *, value_sep, kv_sep)` helper + required-key validation.~~ (`run_dir.py`)
- ~~Thread `run_name` into `setup_run_dir`; rework `_claim_dir` for post-timestamp `_<n>`.~~
- ~~`--experiments_dir` flag → `setup_run_dir(experiments_root=...)`.~~
- ~~Replace hardcoded W&B `run.name` with `run_name`.~~
- ~~Spectrum → `<run_dir>/spectrum.p`; delete `--artifact_suffix`; retire `artifact_stem`.~~
      (`backfill_alpha_wandb.py` left intact with a LEGACY note — it migrates *old*
      runs' JSON stems via the W&B API and gracefully skips new runs.)
- ~~Add `run_name_format` to `configs/cifar3_rholoss.yaml`, `makeblobs_uniform.yaml`,
      the template, and `configs-temp/makeblobs_smoke.yaml`.~~
- ~~Update CLAUDE.md (drop stale `save_labels.py` line + 4-flag CLI; document `run_name_format`).~~
- ~~GPU smoke test + commit.~~ (run dir `20260626_113249_run_MakeBlobs_Noise_Uniform_SGD_lr-0.01_CrossEntropy`;
      W&B name = run_name; config snapshot has `run_name`, no `artifact_stem`; no shared `spectrum/`.)