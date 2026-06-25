# Plan: Spring Cleaning

A merged refactor combining two efforts:
1. **New diagnostics system** (was `plan_new_diagnostics_3.md`) — a dependency-aware, cached, self-registering `Diagnostic` framework.
2. **New nomenclature** (was `plan_new_nomenclature.md`) — unique run identities, a single output tree under `./experiments/`, a readable label cache, and single-file config templates.

This document is the implementation-ready spec. The two source plans remain for the design discussion (the `[[ ]]`/`{{ }}` notes); this plan is what we build from.

---

## 0. Guiding Principles

- **Every run is self-contained.** All outputs of a run live under one directory: `./experiments/run_<timestamp>/`.
- **Collisions are bugs, and bugs are loud.** Any attempt to overwrite an existing output file raises an exception. The *only* exception is the cache (see §3), where a verified hit is a deliberate, silent reuse.
- **One config per run.** A run is driven by a single merged config file, not four mix-and-match files. Sweeps are generated from templates.
- **No new heavyweight dependencies.** Config templating is hand-rolled on `pyyaml`; we do not adopt OmegaConf/Hydra.

---

## 1. Directory Layout

```
./experiments/                         # all run outputs (git-ignored)
    slurm_history/                     # <SLURM_JOB_ID> -> run_<ts> pointers for restart recovery (§9.2)
    run_20260625_131500/
        config.yaml                    # the exact resolved config this run used
        wandb/                         # W&B run files (wandb.init(dir=...))
        logs/                          # file_log outputs from diagnostics
        snapshots/                     # checkpoints, spectral snapshots, NTK, etc.
        labels -> ../../cache/labels/cifar3_train_seed42_labels.pt   # symlink (see §3)
./cache/                               # shared, cross-run cache (git-ignored)
    labels/
        cifar3_train_seed42_labels.pt
./configs-temp/                        # generated sweep configs (git-ignored)
./configs/                             # hand-written single-file configs / templates
```

Add `experiments/`, `cache/`, and `configs-temp/` to `.gitignore`.

---

## 2. Run Identity & Collision Rules

- Each run gets an id: `run_<timestamp>` with a `_<n>` suffix added only on collision, e.g. `run_20260625_131500`, then `run_20260625_131500_1`, `run_20260625_131500_2`, …
    - `timestamp`: `YYYYMMDD_HHMMSS`.
    - **No hash.** The run dir is claimed with an *atomic* `os.mkdir` (or `os.makedirs(..., exist_ok=False)`), which fails if the dir already exists. On `FileExistsError`, increment the `_<n>` suffix and retry until a fresh name is created. This is race-safe across parallel SLURM jobs: two jobs in the same second can't both win a name; the loser bumps to the next `_<n>`.
    - **Consequence — the run dir name is only known at runtime**, after the atomic create; it cannot be precomputed in the submit script the way `get_save_dir.py` does today. Point SLURM's own stdout/stderr at a separate job-id-keyed location (e.g. `logs/slurm/%j.out`); copy or symlink them into the run dir at startup if you want them there.
- A central helper claims the run directory atomically and creates all its subdirs **once** at startup, returning the paths. The atomic create *is* the collision check — no separate exists-then-create (that would have a TOCTOU race between parallel jobs).
- A small write-guard utility is used for all run outputs: *if the target path exists, raise.* This enforces "collisions are bugs" uniformly across logs, snapshots, and config snapshots.
- The resolved config actually used by the run is written to `experiments/run_.../config.yaml` at startup. This is the durable record (W&B also logs it).

---

## 3. Label Cache

Decision: **readable names, no hash, loud on overwrite** (option A from the source notes — label inputs are all short scalars, so the filename can losslessly encode them).

- Cache lives in `./cache/labels/`, shared across runs.
- Filename **losslessly encodes every determining input**: dataset, split, seed, and any transform that changes the labels. E.g. `cifar3_train_seed42_labels.pt`. The determining inputs are an explicit, auditable list in code (e.g. `LABEL_CACHE_KEYS`), not "whatever is in the config dict."
- **Read path:** if the file exists → reuse it silently (this is the cache hit; it is *not* a collision). If absent → compute and write.
- **Write path:** writing uses the §2 write-guard semantics with one twist — `save_labels.py` first checks existence; on a hit it skips the write. It must **never** silently overwrite an existing cache file. If code paths ever attempt to write *different* content to an existing name, that's a naming bug → raise loudly.
- In each run dir, create a **symlink** `experiments/run_.../labels -> ../../cache/labels/<name>.pt` so the run is browsable as self-contained while storage stays shared and `save_labels.py` stops overwriting.
- Caveat (documented, not blocking): symlinks don't survive `tar`/`scp` cleanly. If a run dir is ever archived off-cluster, the link dangles. Mitigation: the cache filename is fully determined by inputs recorded in `config.yaml`, so the link is reconstructible; or hardlink if same-filesystem archival is needed.
- Future note: if a cached artifact ever has inputs that *aren't* short scalars (a transform pipeline, an index list), switch that cache to option B (readable abbreviated name + sidecar `.meta` input-hash, reuse-on-match / raise-on-mismatch). Not needed for labels.

---

## 4. Config System

### 4.1 Single merged config

- A run is driven by **one** YAML file: the concatenation of what used to be the `method`, `data`, `model`, `optim`, and `diagnostics` configs, nested under top-level sections:

```yaml
data:    { ... }
model:   { ... }
optim:   { ... }
method:  { ... }
diagnostics: { ... }   # consumed by create_diagnostics.py (see §5.7)
wandb:   { ... }       # everything wandb.init needs to set up W&B (see below)
```

- The `wandb:` section holds everything `wandb.init` needs — `project`, `entity` (org), and optionally `group`, `tags`, `mode` (`online`/`offline`/`disabled`), etc. At startup `main.py` calls `wandb.init(**config["wandb"], dir=run_dir/"wandb")` — the `dir=` is forced from the run dir per §6.4, not taken from config. Exact key list to be finalized during implementation.
- `main.py` loads this one file. (The old four-`--`flag interface is replaced; see §7 for the migration of `create_diagnostics.py` to read the `diagnostics:` subtree instead of its own file.)

### 4.2 Config templates & generation

A template is a single config with some leaf values set to the sentinel `__REQUIRED__`, marking values that must be supplied at generation time.

`generate_configs(template_path, params_to_vary)`:

- `params_to_vary`: `dict[str, list]` keyed by **dotted paths** into the config, e.g.
  ```python
  params_to_vary = {
    "optim.lr": [1e-3,1e-2,1e-1],
    "model.name": ["LeNet","ResNet"]
  }
  generate_configs("configs/cifar3_base.yaml", params_to_vary)
  ```
  
- **Validation rules (all raise on violation):**
    1. Every dotted key in `params_to_vary` **must exist** in the template (typo protection).
    2. Every key in `params_to_vary` **must currently be `__REQUIRED__`** in the template. Passing a key that already holds a concrete value → raise. (Original rule: "if a key is not null and is passed, raise.")
    3. Every `__REQUIRED__` leaf in the template **must be covered** by `params_to_vary`. An unfilled `__REQUIRED__` after generation → raise. (Original rule: "if a key is null, the generator must be passed it.")
- **Expansion:** full **Cartesian product** over the value lists.
- **Output naming:** each generated file is named `<template_stem>_<k1><v1>_<k2><v2>...yaml`, i.e. the template's name followed by an underscore-separated list of the changing parameters and their values (dotted keys sanitized for filesystem safety). E.g. `cifar3_base_lr0.01_nameResNet.yaml`.
- **Output location:** `./configs-temp/` (git-ignored), for use by the SLURM submission scripts and tests. This on-disk staging is a convenience, not load-bearing — generating in-memory and handing configs straight to the submit script is acceptable. The only hard requirement is that whatever config a run consumes is snapshotted into `experiments/run_.../config.yaml`.
- **Collision check:** generated filenames go through the §2 write-guard — if a target already exists, raise.
- The config a run ultimately uses is **also** snapshotted into its `experiments/run_.../config.yaml` (§2) so W&B and the run dir record exactly what ran.

---

## 5. Diagnostics System

(Carried over from `plan_new_diagnostics_3.md`, integrated with §1–§4.)

### 5.1 `TrainState` (dataclass)
- `epoch: int`, `batch_idx: int`, `total_epochs: int`, `total_batches: int`, `total_steps: int`
- `phase: Phase` — an enum distinguishing the point in the iteration the state was captured (e.g. `PRE_BATCH`, `POST_BATCH`, `VALIDATION`), one value per manager.

**Design rationale (the caching invariant).** `TrainState` must uniquely identify the model/optimizer/data situation any diagnostic could read, so that equal `TrainState` ⇒ identical diagnostics. The minimal correct cache key is `(total_steps, phase)`:
- `total_steps` is the monotonic global step counter. Model weights and optimizer state change exactly once per optimizer step, so equal `total_steps` ⇒ same weights. This is the load-bearing field; `epoch`/`batch_idx` are derivable from it but kept for readable logs.
- `phase` is mandatory because `total_steps` alone does **not** separate pre-batch from post-batch within one step — the weights are updated between them, so without a discriminator they'd wrongly share a cache entry. Each manager stamps its own phase.

Do **not** put values diagnostics *read* (batch tensors, current loss) into `TrainState` — they break `==`, bloat the key, and invert the design. `TrainState` answers *when/where are we*; `shared_context` carries *the data*. Add a field only if a future diagnostic's output depends on something not captured by `(total_steps, phase)`.

### 5.2 `DiagnosticInfo` (dataclass)
- `name: str`, `info: Any`
- May gain an optional `metadata: dict` during implementation (e.g. units, a W&B panel/section hint, scalar vs. histogram vs. image) so `wandb_log`/`file_log` can route output without hard-coding per diagnostic. Kept to `name`/`info` until implementation forces more.

### 5.3 `Diagnostic` (partially abstract)

**Attributes**
- `manager: DiagnosticsManager` — given in constructor; auto-registers self with manager.
- `log_path: str | None` — given in constructor; defaults to `None`; `file_log` is a no-op if unset. **Composed from the run dir** (§7): top-level diagnostics receive `run_dir/logs/...`.
- `should_run: Callable -> bool` — given in constructor; defaults to `lambda: True`.
- `last_run_state: TrainState | None` — initially `None`.
- `last_run_diagnostic: DiagnosticInfo | None` — initially `None`.

**Methods**
- `get_state() -> TrainState` — returns `manager.current_state`.
- `get_context() -> dict` — returns `manager.shared_context`.
- `_run() -> DiagnosticInfo` — **abstract.** Takes no args. Child calls `dep.run()` for each dependency first, then computes and returns a `DiagnosticInfo`. `dep.run()` caches by state, so calling it here is cheap if the dep already ran this step.
- `run() -> DiagnosticInfo` — if `get_state() == last_run_state`, returns cached `last_run_diagnostic`; otherwise calls `_run()`, updates `last_run_state`/`last_run_diagnostic`, returns result. Cache is keyed on state only — diagnostics must not depend on per-call args that vary independently of state; pass such values through `shared_context`.
- `conditional_run()` — calls `should_run(get_state())`; if `True`, calls `run()`. Used by the manager.
- `wandb_log(infos: List[DiagnosticInfo])` — logs `last_run_diagnostic` with necessary elements of `last_run_state` to W&B.
- `file_log(infos: List[DiagnosticInfo])` — logs `last_run_diagnostic` with necessary elements of `last_run_state` to `log_path`.
- `log()` — calls `wandb_log([last_run_diagnostic])` and `file_log([last_run_diagnostic])`.
- `__eq__` — raises `NotImplementedError` by default; children must override if they may be deduplicated in `create_diagnostics.py`.

### 5.4 `DiagnosticsManager`

**Attributes**
- `diagnostics: List[Diagnostic]` — top-level diagnostics only; dependency diagnostics are not registered here.
- `current_state: TrainState`
- `should_run: bool` — master kill-switch for all diagnostics in this manager.
- `shared_context: dict`

**Methods**
- `_update_state(state)` — sets `current_state`.
- `_update_shared_context(**kwargs)` — merges kwargs into `shared_context`.
- `run_diagnostics(state, *, **kwargs)` — runs `_update_state`, `_update_shared_context`, then `conditional_run()` on each registered diagnostic, then `_log_diagnostics()`.
- `_log_diagnostics()` — calls `log()` on each registered diagnostic.

*Multiple managers exist for different training phases (pre-batch, post-batch, validation, etc.).*

### 5.5 `DiagnosticsBuilder`

**Attributes**
- `all_diagnostics: defaultdict(list)` — keyed by diagnostic class; values are instance lists.

**Methods**
```python
def fetch_duplicate_diagnostic(self, diagnostic) -> Diagnostic | None:
    # WARNING: `==` calls Diagnostic.__eq__, which raises NotImplementedError unless the class overrides it.
    matches = [x for x in self.all_diagnostics[type(diagnostic)] if x == diagnostic]
    if not matches:
        return None
    elif len(matches) > 1:
        raise ValueError(f"Multiple identical diagnostics of type {type(diagnostic).__name__}")
    else:
        return matches[0]

def build(self, diagnostic_class, *args, **kwargs) -> Diagnostic:
    new_diagnostic = diagnostic_class(*args, **kwargs)
    duplicate = self.fetch_duplicate_diagnostic(new_diagnostic)
    if duplicate:
        return duplicate
    self.all_diagnostics[diagnostic_class].append(new_diagnostic)
    return new_diagnostic
```

### 5.6 `create_diagnostics.py`
- Initializes a `DiagnosticsBuilder` and the appropriate `DiagnosticsManager`s.
- **Reads the `diagnostics:` subtree of the single merged config** (§4.1), *not* a standalone diagnostics file. Keys are `Diagnostic` class names in scope.
- Calls `DiagnosticsBuilder.build()` per top-level diagnostic; shared dependencies built internally are deduplicated via `__eq__`.
- Registers top-level diagnostics with the appropriate manager; dependency diagnostics are not added to any manager's list.
- Receives the run dir (§7) and composes each top-level diagnostic's `log_path` under `run_dir/logs/` (and snapshot paths under `run_dir/snapshots/`).

### 5.7 Diagnostics config (the `diagnostics:` section)
```yaml
diagnostics:
    logging_defaults:
        log_interval: logarithmic
        save_init: 5
        save_freq: 4
    diagnostics:
        GradNorm:                  # example
            logging:               # optional per-diagnostic override of defaults
                log_interval: ...
            params:
                norm: 2            # kwargs passed to GradNorm's constructor
```

(Additional keys may be added during implementation as needed.)

---

## 6. Reconciliation Points (where the two plans touch)

These are the seams that must be built deliberately or the merge silently fails:

1. **Cache vs. collision rule.** The cache (§3) is explicitly exempt from "collisions raise": a verified hit is silent reuse. Only unkeyed run outputs get the write-guard. Both rules coexist only because they're scoped this way.
2. **Config layer ownership.** `create_diagnostics.py` must read the `diagnostics:` subtree (§5.6), not its own standalone file — from day one, never a separate file. This is why Phase 3 (single config) must land before Phase 5 (diagnostics): building diagnostics against a standalone YAML first would break when the config merge lands.
3. **Run-dir injection.** The run dir (§2) must be threaded into the diagnostics builder/manager so `log_path` and snapshot paths land under `experiments/run_.../`. Without this, diagnostics scatter output outside the run tree — defeating the whole nomenclature goal.
4. **W&B dir.** `wandb.init(dir=run_dir/wandb)` must be set at startup so W&B files live under the run dir.

---

## 7. Implementation

### 7.1 Style
- Read https://labs.acme.byu.edu/StudentResources/CodeQuality/CodeQuality.html, and follow that style.
- The code should pass a basic linter like flake8.
- Don't be excessively verbose, but be precise and clear. Code should be human-readable.
- Don't add unnecessary/unenlighening comments. Use whitespace in a way that enhances readability without introducing fluff.
- Anytime you need to deviate from our plan, let me know, and ask questions where appropriate! Do not assume that I will agree with you about non-trivial things.

### 7.2 Phases

Ordered so each phase leaves the repo runnable.

**Phase 1 — Output tree & run identity (§1, §2)**
- [x] ~~Add `experiments/`, `cache/`, `configs-temp/` to `.gitignore`.~~
- [x] ~~Run-id + run-dir creation helper (timestamp + `_<n>` suffix on collision per §2 — **no hash**; subdirs; atomic raise-on-exists).~~ (`run_dir.py: setup_run_dir`)
- [x] ~~Write-guard utility (raise if target exists) used by all run-output writes.~~ (`run_dir.py: write_guard`, plus `atomic_save` for the rolling checkpoint)
- [x] ~~Point W&B (`wandb.init(dir=...)`) and run-config snapshot at the run dir.~~ (`main.py`)
- [x] ~~Resume support reconciled with the new scheme (§9).~~ (`main.py: _configure_resume_state`; restart via `slurm_history` pointer, extension via fork+copy)

**Phase 2 — Label cache (§3)**
- [ ] `LABEL_CACHE_KEYS` explicit determining-input list + readable filename builder.
- [ ] `save_labels.py`: read-hit reuses silently; write never overwrites (raise on different content for same name).
- [ ] Symlink the cache file into the run dir.

**Phase 3 — Single merged config (§4.1)**
- [ ] `main.py` loads one merged YAML with `data/model/optim/method/diagnostics` sections.
- [ ] Convert existing four-file configs into merged single files.
- [ ] Move resume inputs into the merged config's `resume:` section (§9.4); retire `training_opt.resume_run_path`/`resume`/`additional_epochs`.

**Phase 4 — Config templates (§4.2)**
- [ ] `__REQUIRED__` sentinel + the three validation rules (raise on each).
- [ ] `generate_configs` with dotted keys, Cartesian product, `<template>_<k><v>...` naming, output to `configs-temp/`, write-guard.
- [ ] Update SLURM submission scripts to generate + consume templated configs.

**Phase 5 — Diagnostics framework (§5)**
- [ ] `TrainState`, `DiagnosticInfo`, `Diagnostic`, `DiagnosticsManager`, `DiagnosticsBuilder`.
- [ ] `create_diagnostics.py` reading the `diagnostics:` subtree and receiving the run dir.
- [ ] Port existing diagnostics (spectral snapshots, NTK, linear probes, checkpointing) onto the new `Diagnostic` base with `__eq__` where dedup is possible.
- [ ] Ensure all diagnostic parameters are included correctly in the single merged config from Phase 3.

**Phase 6 — Wiring & cleanup (§6)**
- [ ] Thread run dir into the diagnostics builder/manager; compose `log_path`/snapshot paths under it.
- [ ] Remove the old `./exp/` path scheme and the standalone diagnostics-config plumbing.

### 7.3 Testing
To test at each phase, use the command `pgpujob`, which makes a preemptible `salloc` with gpu. Run tests like those in `run_*.sh` and `slurm_run_*.py`. Make commits after each phase on the `spring_cleaning` branch. Do not make commits to any other branch, do not change any other code, do not pass go, do not collect $200. Do not yet push or interact with the git remote.

---

## 8. Resolved Decisions

- **Template-param filename uniqueness (resolved):** when two dotted keys share a leaf name (e.g. `optim.lr` and `sched.lr`), the `<k><v>` filename fragment uses the sanitized *full dotted path*, not just the leaf, to stay unique.
- **Phase ordering (resolved):** Phase 3 (single config) lands before Phase 5 (diagnostics), since §6.2 requires diagnostics to read the merged config's subtree from the start.

---

## 9. Resume

There are **two** distinct things we call "resume," and they want **different** mechanisms. Conflating them is what makes this hard; separating them is what makes it clean.

- **Restart (preemption / crash recovery).** A job is interrupted (preemptible SLURM, node failure) and requeued. Conceptually this is the **same run** with the **same config** — it just got interrupted. The right behavior is to land back in the *same* run dir and pick up where it left off, transparently. Forking a new dir on every preemption would litter the tree and fragment one logical run across many dirs.
- **Extension (deliberate continuation).** A run finished its configured `num_epochs` (say 300) and we now decide to push it to 600 and keep training from the last checkpoint. This is a **new logical run** that *branches off* a finished one with a *changed* config. We fork a new dir and **copy the parent's contents into it** at fork time, so the extension is a complete, standalone run rather than something that depends on tracking a parent.

Both must respect §0 ("every run is self-contained"; "collisions are loud"), but they respect it in different ways.

### 9.1 Why the old mechanism can't survive unchanged

Today's resume (`main.py`: `_configure_resume_state`) points the run *back at its original output directory* and writes new outputs on top of the old ones — `checkpoint.pth.tar` is silently overwritten, the logger appends. For **restart** that instinct is *right* (it's the same run), but the silent-overwrite must be made safe against the write-guard. For **extension** that instinct is *wrong* (it clobbers a finished run's record). Hence the split below.

### 9.2 Restart (preemption / crash recovery): same dir, made safe

A restart re-enters the **same** run dir `R` and continues. It is self-contained by construction (one dir for the whole run's life). The two problems to solve are *finding* `R` after a requeue and *not* tripping the write-guard.

**Finding the dir across requeues.** The run dir name is only known at runtime (§2), so the requeued job can't recompute it. When a run dir is first created, also create a stable pointer keyed by the SLURM job id:

```
experiments/slurm_history/<SLURM_JOB_ID> -> run_<timestamp>     # symlink
```

A preemptible job uses `#SBATCH --requeue`, so the requeued job keeps the **same** `SLURM_JOB_ID` (verified against `man sbatch`: under `--requeue`, "the batch script is initiated from its beginning with the same job ID"). The `SLURM_RESTART_COUNT` env var tracks how many times the job has been requeued, so a restart can additionally be detected even without the pointer. Startup logic: if `experiments/slurm_history/$SLURM_JOB_ID` exists → this is a restart; reuse the pointed-to dir, load its latest checkpoint, reattach the same W&B id, continue. If absent → fresh run; create the dir and the pointer. (Outside SLURM, no pointer is made and every launch is fresh — restart is a cluster concern.)

**Not tripping the write-guard.** Distinguish two checkpoint artifacts in `snapshots/`:
- **Immutable per-step/epoch snapshots** (e.g. `snapshots/ckpt_step000123.pth`) — write-guarded, never overwritten.
- **A rolling "latest" checkpoint** `snapshots/checkpoint.pth.tar` — explicitly exempt from the write-guard (declared like §3's cache exemption), written via *atomic replace* (write temp + `os.replace`). It advances; it is never half-written.

On restart we load `checkpoint.pth.tar`, and any immutable snapshot whose step is *beyond* the loaded checkpoint is impossible (the rolling file is written no earlier than its snapshot), so no immutable file is ever overwritten. Config is unchanged; `num_epochs` is unchanged; the W&B run continues via `id=<run's own id>, resume="allow"`.

### 9.3 Extension (deliberate continuation): fork a new dir

An extension is requested explicitly (config, §9.4), pointing at a finished parent dir `P` and supplying a larger epoch budget. It claims a **new** dir `C = run_<timestamp>` (atomic, per §2) and makes `C` a **full copy** of `P`, then continues training inside `C`. `P` is never modified, so the write-guard stays universal with **no** exemption:

1. **Claim a fresh child dir** `C` exactly as a normal run.
2. **Copy `P` into `C`.** Deep-copy the parent's contents (config, logs, snapshots, the `wandb/` dir) into `C` so `C` is a complete standalone run on disk. The copy does **not** go through the write-guard (it is populating a brand-new dir from a trusted source, not overwriting live run output); after the copy, normal write-guarded training resumes and will *append* new immutable snapshots and *advance* the rolling checkpoint within `C`.
   - Exclude the parent's `labels` symlink from a naive copy and re-create it fresh in `C` (per §3), so it points at the shared cache rather than becoming a dangling/parent-relative link.
3. **Record provenance (metadata only):** `resume.from: <P>` in `C/config.yaml` and a `C/resumed_from -> P` symlink. This is non-load-bearing — `C` is already complete; the link just records lineage.
4. **Read the checkpoint** from `C/snapshots/checkpoint.pth.tar` (now local after the copy; overridable). Supplies model/optimizer/scheduler state and the epoch reached.
5. **Reattach W&B:** read the parent's W&B run id (from `C/config.yaml`'s recorded `wandb_run_id`, copied from `P`) and `wandb.init(..., id=<parent_id>, resume="must", dir=C/wandb)`. W&B sees one continuous run; new local files land under `C`.
6. **Epoch target:** `additional_epochs` (if given) sets `num_epochs = checkpoint_epoch + additional_epochs`; else the config's absolute `num_epochs` is used and must exceed the checkpoint epoch (raise otherwise).

This makes an extension a complete, standalone run: it can be browsed, archived, or further extended without reference to `P`. The cost is duplicated checkpoint/snapshot storage per extension, which is acceptable given extensions are deliberate and infrequent.

**Write-guard exemptions (resolved).** Exactly two exist across the whole system: the §3 label cache, and the rolling `checkpoint.pth.tar` in §9.2 (restart). The latter is justified because that "latest" file is *defined* to advance, not a collision. The extension copy in §9.3 needs no exemption — it only writes into a fresh dir.

### 9.4 Config surface

In the four-flag era (before Phase 3), the existing `training_opt.resume_run_path` / `resume` / `additional_epochs` keys keep working but are rewired to the §9.3 fork+copy behavior; restart (§9.2) is driven by the `slurm_history` pointer and needs no config keys. From Phase 3 on, resume inputs move into a top-level `resume:` section of the merged config:

```yaml
resume:
    from: ./experiments/run_20260625_131500   # extension parent dir; omit/null = not an extension
    checkpoint: null                           # optional explicit checkpoint path; default = <from>/snapshots/checkpoint.pth.tar
    additional_epochs: 300                     # optional; if null, use absolute optim.num_epochs (must exceed checkpoint epoch)
    allow_restart: true                        # opt into §9.2 slurm_history restart recovery (default true under SLURM)
```

If `resume.from` is null/absent and no `by_job` pointer matches, the run is an ordinary fresh run.

### 9.5 Phase placement

- **Phase 1:** run-dir helper supports both paths — (a) on creation, also write the `experiments/slurm_history/<jobid>` pointer and, on restart, detect+reuse the pointed-to dir (§9.2); (b) on extension, claim a fresh child dir, copy the parent into it, and write the `resumed_from` symlink (§9.3). Add the rolling-checkpoint write-guard exemption + atomic-replace write. Rewire `main.py`'s `_configure_resume_state` to read checkpoint/epoch/W&B id from the source and target the correct dir. No four-flag config changes yet.
- **Phase 3:** introduce the `resume:` config section (§9.4) and retire the `training_opt.resume_*` keys.
- **Phase 6:** ensure diagnostics checkpoint output writes the rolling `snapshots/checkpoint.pth.tar` (atomic replace) plus immutable per-step snapshots, so both restart and extension have a well-defined source.
