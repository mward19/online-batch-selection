2. main.py:178-179 — the --wandb_project CLI flag overrides wandb.project if passed. [[Eliminate this flag. The config should determine the project]]


Read line 124 of SelectionMethod.py. I want this whole big dictionary unpacking to happen in the diagnostics code, not in selection method. I propose that instead of
```python
self.diagnostics = create_diagnostics(config.get('diagnostics', {}), diagnostics_resources) 
```
we do
```python
self.diagnostics = create_diagnostics(
    self,
    diagnostics_resources
)
```
where `diagnostics_resources` is a small dictionary like
```python
diagnostics_resources = {
    'project_root': project_root,
    'dataset_name': dataset_name,
    'model_name': model_name,
}
```

The config is an attribute of `self`, so I don't think it needs to be passed as `config.get('diagnostics', {})`, right?

<!-- # [[What if instead we didn't make diagnostics_resources and just passed `self` (SelectionMethod)? And create_diagnostics could also receive **kwargs and save those to the context too in some sense]] -->