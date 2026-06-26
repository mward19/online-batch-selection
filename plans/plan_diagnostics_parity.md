# Plan: Diagnostics parity audit — deviations from `plan_spring_cleaning.md`

Audit of the implemented code against `plans/completed/plan_spring_cleaning.md`
("to the tee, except where clarified by `plan_method_utils_cleanup.md`"). Records
where the code diverges and proposes what (if anything) to do. **Nothing here is
implemented yet — for review.**

Branch: `spring-cleaning`. No remote push.

Leave decisions as `[[ ... ]]` notes; I'll answer in `{{ ... }}` or fold into the
text, then implement only what you approve.

---

## A. Matches the plan (no action)
§1–§4 and §9 infrastructure all verified present and faithful:
- Run dir: atomic `_<n>`-suffix claim, subdirs, `write_guard`, `atomic_save`
  (`run_dir.py`).
- Resume: `slurm_history/<jobid>` restart pointer; extension fork + `_copy_run_contents`
  + `resumed_from` link (`run_dir.py: setup_run_dir`).
- Label cache: `LABEL_CACHE_KEYS`, readable filename, `cache/labels/`, run-dir
  `labels` symlink (`generate_noise.py`).
- Single `--config` CLI; `generate_configs` templates (dotted keys, Cartesian,
  write-guard); `wandb.init(dir=run_dir)`.

## B. Deviations already documented & accepted
- **§5.8.4 NTK/probe/param/weight ported as single leaves** (no `NTKKernel`/
  `PenultimateFeatures` deps, no `NTKSpectrum`/`Alignment`/`Distance` split).
  Written up in the §5.8.4 "deviation, accepted" note — zero compute sharing, so
  the split is pure risk. **No action.**

## C. Deviations authorized by `plan_method_utils_cleanup.md`
- `get_context()` returns merged `static_context`+`shared_context` (raises on
  collision), not bare `shared_context` (§5.3).
- `create_diagnostics(diagnostics_config, resources)`; `DiagnosticsRunContext`
  deleted; engines live in the diagnostics package (§5.8.7).
- **No action** beyond noting these supersede the spring-cleaning text.

---

## D. Undocumented deviations — decisions needed

### D1. Plan-wording fixes (code is right, plan text is stale) — low effort
1. **§5.3 "auto-registers self with manager" is false** and contradicts §5.4/§5.6/
   §5.8.4 (deps must *not* be registered). Code: `create_diagnostics` registers
   only top-level leaves; `Diagnostic.__init__` does not self-register. **Proposed:**
   amend the completed plan's §5.3 to say "registration is explicit in
   `create_diagnostics`; dependency diagnostics are never registered."
2. **§5.4 `_log_diagnostics()` "log() on each registered diagnostic"** — code logs
   only diagnostics whose `last_run_state == current_state` (i.e. that actually ran
   this step), so schedule-gated diagnostics don't re-emit stale values. **Proposed:**
   amend §5.4 to state the state-guard.
   - {{Both are correctness-preserving; I recommend just fixing the plan text.
     Approve?}}

### D2. Functional regressions (behavior the old system had, now missing)
These were lost when `SnapshotManager`/`DiagnosticsLogger` were deleted in Phase 6
and not re-created as leaves. Each is a real capability drop, not just naming.

3. **Local snapshot-history payload (`SnapshotSave`) — MISSING.**
   - Plan: §5.8.4 lists `SnapshotSave`; §5.8.6/§9.5 say the SnapshotManager
     `snapshots`/`steps` history payload moves under `run_dir/snapshots/`.
   - Now: nothing writes it. The old `local_snapshots` `.p` history and the
     `local_points_selected` per-epoch `.npy` mask are gone.
   - **Proposed fix:** a `SnapshotSave` epoch-end/finalize leaf that accumulates the
     per-log-step snapshot rows (it can depend on the existing `ForwardPass`/
     `PerSampleLossError` so it adds no recompute) and writes
     `run_dir/snapshots/<artifact_stem>.p` at `finalize()`. Add `SelectedPoints`'
     per-epoch mask `.npy` save behind the same leaf or a flag.
   - {{Do you still use these local `.p`/`.npy` artifacts (vs. reading from W&B)?
     If not, I'd rather formally drop them in the plan than rebuild them. Which?}}

     [[Would these snapshots be necessary to resume a run?]]

4. **Immutable per-step checkpoints — MISSING.**
   - Plan: §9.2 + §9.5-Phase-6 specify write-guarded `snapshots/ckpt_step<NNN>.pth`
     alongside the rolling `checkpoint.pth.tar`.
   - Now: `Checkpoint` writes only the rolling `checkpoint.pth.tar` + `model_best.pth.tar`.
   - **Proposed fix:** in the `Checkpoint` leaf, also write an immutable
     `ckpt_step{total_steps}.pth` (via `write_guard`) on its schedule.
   - {{This adds real disk cost per logged step. Do you actually want the immutable
     per-step history, or is rolling + best sufficient? I lean "rolling + best is
     enough; drop the immutable-snapshot requirement from the plan" unless you have
     a use (e.g. reconstructing training trajectories).}}

     [[No. Just make sure we log the last model (to be able to resume), and the best model (for fun).]]

5. **Wall-clock telemetry not logged.**
   - Plan: §5.8.3 lists `total_time`/`time_this_epoch` in `shared_context`.
   - Now: not passed by `SelectionMethod` and no leaf logs it. (Still saved inside
     `checkpoint_state`, so resume timing is intact; only the W&B/file metric is gone.)
   - **Proposed fix:** a tiny `Timing` leaf logging `total_time`/`time_this_epoch`,
     fed via post-batch `shared_context`.
   - {{Low effort. Want it back as a metric, or fine to leave timing in checkpoints
     only?}} [[Yes, make the Timing `Diagnostic` object and integrate it where appropriate.]]

### D3. Minor / cosmetic
6. **`wstar/what/bayes` W&B summary constants dropped.** Old `DiagnosticsLogger`
   wrote them to `wandb.summary`; no leaf reproduces it. The values still flow into
   the resources dict (`wstar_test_acc`, etc.). **Proposed:** fold into the
   `SnapshotSave`/a `finalize()` hook if D3 isn't otherwise built; else a 3-line
   `wandb.summary` write at run start. {{Keep or drop?}} [[Keep these, but they need to move.]]
7. **Leaf naming differs from the plan** (`TrueLabelTrainLoss` vs `TrainLossTrueLabels`,
   `LinearProbe` vs `LinearProbeAcc`, `NTK` vs `NTKSpectrum/...`; "noisy-label"
   metrics are just `TrainLoss`/`TrainAcc` on loader labels). Behavior-equivalent.
   **Proposed:** leave as-is; not worth churning configs. {{OK?}} [[ok]]

---

## E. Suggested checklist (pending your D answers)
- [ ] D1: amend §5.3 + §5.4 wording in the completed plan (always safe).
- [ ] D2.3: build `SnapshotSave` (or formally drop in plan).
- [ ] D2.4: immutable per-step checkpoints in `Checkpoint` (or formally drop).
- [ ] D2.5: `Timing` leaf (or formally drop).
- [ ] D3.6/D3.7: decide keep/drop.
- [ ] If any leaf added: migrate the example configs' `diagnostics:` to enable it;
      GPU smoke test on `configs-temp/makeblobs_smoke.yaml`; commit.
- [ ] Move this plan to `plans/completed/` when resolved.

## F. My recommendation
The honest split: **D1 is a doc fix (do it).** For **D2**, my hunch is the project
moved to W&B-centric logging, so the local `.p`/`.npy` history (D2.3) and immutable
per-step checkpoints (D2.4) may be genuinely obsolete — in which case the right
move is to **amend the plan to drop them** rather than rebuild. **D2.5 (timing)**
and **D3.6 (teacher-acc summaries)** are cheap to restore if you want the metrics.
Tell me which of D2/D3 are still wanted and I'll either build or formally retire them.
