# Adopt OmegaConf in `workflow_torch/` for structured, self-describing configs

## Context

**Question asked:** Would Hydra + OmegaConf simplify the codebase?

**Answer:** OmegaConf — yes, a contained, low-risk win. Hydra — no (overkill for a
3-script active surface, no sweep/W&B workflow, and its cwd-rewrite + `@hydra.main()`
decorator fight the notebook-heavy and GCS relative-path workflow). **Decision: adopt
OmegaConf only, scoped to the active `workflow_torch/` scripts.** Legacy
`train_torch/`, `train_generic/`, `train_gcs/`, and notebook configs are left untouched.

**What's painful today** (verified in code):

1. **Duplicated unpacking** — the same ~40-line `cfg.get(...)` block is repeated in
   [train.py:104-144](workflow_torch/train.py#L104-L144),
   [resume.py:123-144](workflow_torch/resume.py#L123-L144),
   [evaluate.py:98-127](workflow_torch/evaluate.py#L98-L127). Defaults are restated in
   every file; a typo'd YAML key is silently ignored.
2. **35 lines of hand-written config logging** per script
   ([train.py:201-238](workflow_torch/train.py#L201-L238)).
3. **Correctness footgun** — architecture/slice params must be kept in sync *by hand*
   between the training config and `config_resume.yaml`/`config_eval.yaml`.
   `resume.py` re-declares `n_modes`/`hidden_channels`/slices and they *must match the
   checkpoint or loading fails*; `evaluate.py` works around it by re-reading the training
   config via `eval_cfg['train_config']`. The checkpoint already stores a copy of the raw
   YAML ([training_utils.py:158](loc_libs/training_utils.py#L158)) but nothing loads it back.

**Intended outcome:** define the config schema *once* (typed, with defaults), validate at
load time, dump it in one call, and make resume/eval *load the checkpoint's own resolved
config and merge a small override* — removing the manual-sync footgun. No change to the
`--config FILE` CLI or the YAML files users already write.

## Approach

### 1. Add the dependency
- [pyproject.toml](pyproject.toml#L10): add `"omegaconf>=2.3"` to `dependencies`.

### 2. New shared schema module: `workflow_torch/_config.py`
Lives next to `_model.py` (scripts already add the script dir to `sys.path`). Defines
`@dataclass` schemas whose field names/defaults mirror the **current** `cfg.get(...)`
defaults exactly (no behavior change):
- Nested: `NormalizationCfg` (`name='none'`, `alpha`), `DownsampleCfg`, `LoaderCfg`
  (`num_workers=0`, `prefetch_factor=2`, `locality_block`), `CausalCfg`.
- `TrainConfig` — all train keys (data_source, odir/model_dir, input_dir/input_files,
  lazy, nevents_in, start_event, choose_ntau, n_epochs, batch_size, hidden_channels,
  n_modes, train_ratio, xy/eta/tau slices as `list`, lr, weight_decay, checkpoint_freq,
  min_save_epoch, save_last, eval_interval, test_batch_sizes, log_directly, GCS keys,
  model_type, causal/normalization/downsample/loader sub-configs).
- `ResumeConfig` — train fields **plus** resume-specific (`total_epochs`,
  `resume_model_source`, `resume_model_dir`/`model_dir`, `checkpoint_type`,
  `checkpoint_epoch`, `local_model_cache`).
- `EvalConfig` — eval-specific fields (train_config, checkpoint_type/epoch,
  use_train_split, nevents, active_threshold, eval_batch_size, metric_backend, odir,
  save_figures, local_data_dir, overwrite_cache, eval_input_files, GCS keys).
- Helpers:
  - `load_config(path, Schema, *, overrides=None)` →
    `OmegaConf.merge(OmegaConf.structured(Schema), OmegaConf.load(path), overrides or {})`.
  - Keep the existing `_slice(val)` helper here (slices stay as `[start, stop]` lists in
    YAML; converted to `slice` at point of use, exactly as today). Move the three local
    `_slice` defs out of the scripts and import this one.

Note: structured configs reject unknown keys by default — this is the desired
typo-catch. The schema must therefore cover **every** key the current YAMLs use; verify
by running each script once (see Verification).

### 3. `workflow_torch/train.py`
- Replace `yaml.safe_load` + the 40-line unpack block ([L55-56](workflow_torch/train.py#L55-L56),
  [L104-144](workflow_torch/train.py#L104-L144)) with `cfg = load_config(args.config, TrainConfig)`.
  Keep the existing local variable names by reading from `cfg` (still typed/centralized).
- Replace the 35-line manual log dump ([L201-238](workflow_torch/train.py#L201-L238)) with
  `log(OmegaConf.to_yaml(cfg))`, retaining only the few *derived* log lines (resolved
  source paths, inferred HDF5/ROOT type).
- In the save step, write the **resolved** config instead of copying the raw file:
  `OmegaConf.save(cfg, Path(model_dir) / 'config.yaml')`. Adjust
  `save_last_checkpoint` call ([train.py ~L351](workflow_torch/train.py#L351)) /
  [training_utils.py:158](loc_libs/training_utils.py#L158) so the self-contained config is
  the resolved one (drop the `shutil.copy2`, or pass the already-saved path).

### 4. `workflow_torch/resume.py` — load checkpoint config, merge override
Reorder so the saved config drives architecture/slices instead of restating them:
1. Read the small resume YAML first only to locate the checkpoint
   (`resume_model_source`, `resume_model_dir`/`model_dir`, `checkpoint_type/epoch`).
2. Resolve/download the checkpoint dir (existing logic, [resume.py:250-276](workflow_torch/resume.py#L250-L276)).
3. `base = OmegaConf.load(ckpt_dir / 'config.yaml')` (the resolved train config saved in §3).
4. `cfg = OmegaConf.merge(OmegaConf.structured(ResumeConfig), base, OmegaConf.load(args.config))`.
- Result: `config_resume.yaml` shrinks to just the resume knobs — no more hand-copied
  `n_modes`/`hidden_channels`/slices, removing the mismatch footgun.

### 5. `workflow_torch/evaluate.py`
- Load `eval_cfg = load_config(args.config, EvalConfig)`; validate the training side via
  `train_cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(train_config))`.
- Default `train_config` to `{model_dir}/config.yaml` so eval auto-uses the checkpoint's
  own resolved config (explicit path still honored). Keep the two-config structure;
  just swap loaders + add validation, replacing [evaluate.py:98-127](workflow_torch/evaluate.py#L98-L127).

## Critical files
- `pyproject.toml` — add `omegaconf` dep.
- `workflow_torch/_config.py` — **new**, the single source of truth for schema + `load_config` + `_slice`.
- `workflow_torch/train.py` — use schema, one-call log dump, save resolved config.
- `workflow_torch/resume.py` — load checkpoint config + merge override (key correctness fix).
- `workflow_torch/evaluate.py` — use schema, default `train_config` to checkpoint config.
- `loc_libs/training_utils.py` — `save_last_checkpoint` writes resolved config (minor).

## Reuse / no-change notes
- Existing YAML files load unchanged (OmegaConf reads the same YAML); the merge only
  layers in defaults + type checks.
- `build_source`/`build_dataloaders`/`build_model`/`Normalizer` signatures are untouched —
  they still receive plain values; OmegaConf containers behave like dicts/lists
  (`OmegaConf.to_container(cfg, resolve=True)` is available if any callee needs a pure dict).
- Notebooks and legacy dirs are out of scope and unchanged.

## Verification
1. **Train (smoke):** `cd workflow_torch && python train.py --config config.yaml`
   (the bundled example uses `nevents_in: 10`, `n_epochs: 1`). Confirm: trains to
   completion; `checkpoints/config.yaml` is the **resolved** OmegaConf dump; the training
   log contains the single `OmegaConf.to_yaml` block.
2. **Resume:** point `config_resume.yaml` at that checkpoint with only resume knobs
   (`total_epochs`, `checkpoint_type: last`) — **omit** `n_modes`/`hidden_channels`/slices —
   then `python resume.py --config config_resume.yaml`. Confirm it loads arch from the
   saved config and resumes without a mismatch error.
3. **Evaluate:** `python evaluate.py --config config_eval.yaml` (with `train_config`
   unset → defaults to the checkpoint's `config.yaml`). Confirm metrics/figures produced.
4. **Schema validation (negative test):** add a misspelled key or wrong-typed value to a
   YAML and confirm OmegaConf raises at load — demonstrating the typo-catch win that the
   old `cfg.get()` silently swallowed.
5. **Diff sanity:** confirm net line reduction in the three scripts (removal of the
   repeated unpack + manual log blocks) and that no production code path imports Hydra.
