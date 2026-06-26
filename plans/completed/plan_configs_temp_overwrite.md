# Plan: allow overwriting in `configs-temp/`

## Motivation

`generate_configs.py` writes each generated config to `./configs-temp/` through
`write_guard` (`generate_configs.py:100`), which raises `FileExistsError` if the
target already exists. Regenerating a template into the same dir therefore
crashes. `configs-temp/` is a scratch/derived dir, not a run output, so
overwriting there is fine — the write-guard's "collisions are bugs" rule should
not apply to it.

## Change

~~Implemented.~~

In `generate_configs.generate_configs`, drop the `write_guard(out_path)` call at
`generate_configs.py:100` and overwrite freely. Print a one-line warning when an
existing file is being clobbered.

Replace:

```python
        out_path = os.path.join(out_dir, f"{template_stem}_{'_'.join(fragments)}.yaml")
        write_guard(out_path)
        with open(out_path, "w") as f:
```

with:

```python
        out_path = os.path.join(out_dir, f"{template_stem}_{'_'.join(fragments)}.yaml")
        if os.path.exists(out_path):
            print(f"Warning: overwriting existing config {out_path}")
        with open(out_path, "w") as f:
```

Then remove the now-unused `from run_dir import write_guard` import
(`generate_configs.py:16`) **only if** `write_guard` is used nowhere else in the
file (it currently is not).

## Notes

- `write_guard` itself and its use for real run outputs are untouched.
- The module docstring (`generate_configs.py:7`) mentions writing "through the §2
  write-guard"; update that phrase to reflect that `configs-temp/` is exempt and
  overwriting is allowed.
