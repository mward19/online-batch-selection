"""Template-driven config generation.

A template is a single merged config with some leaf values set to the
sentinel ``__REQUIRED__``, marking values that must be supplied at generation
time. ``generate_configs`` fills those leaves over the Cartesian product of the
supplied value lists, writing one merged config per combination to
``./.configs-temp/``. That directory is scratch/derived output and is exempt from
the write-guard: existing files are overwritten (with a warning).
"""

import copy
import itertools
import os

import yaml

REQUIRED = "__REQUIRED__"
CONFIGS_TEMP_DIR = ".configs-temp"


def _iter_required_paths(node, prefix=""):
    """Yield the dotted path to every ``__REQUIRED__`` leaf in a nested dict."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else key
            yield from _iter_required_paths(value, child)
    elif node == REQUIRED:
        yield prefix


def _get_dotted(node, dotted):
    cur = node
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _set_dotted(node, dotted, value):
    parts = dotted.split(".")
    cur = node
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def _sanitize(text):
    return str(text).replace("/", "-").replace(" ", "").replace("[", "").replace("]", "")


def _filename_fragment(dotted_key, value):
    # Use the full dotted path (dots -> '-') so two keys sharing a leaf name
    # (e.g. optim.lr vs sched.lr) produce distinct fragments.
    return f"{_sanitize(dotted_key.replace('.', '-'))}{_sanitize(value)}"


def _validate(template, params_to_vary):
    required_leaves = set(_iter_required_paths(template))
    for dotted in params_to_vary:
        try:
            current = _get_dotted(template, dotted)
        except KeyError:
            raise KeyError(f"params_to_vary key '{dotted}' is not present in the template.")
        if current != REQUIRED:
            raise ValueError(
                f"params_to_vary key '{dotted}' must be '{REQUIRED}' in the template, "
                f"but holds a concrete value ({current!r})."
            )
    uncovered = required_leaves - set(params_to_vary)
    if uncovered:
        raise ValueError(
            f"Template has unfilled {REQUIRED} leaves not covered by params_to_vary: "
            f"{sorted(uncovered)}"
        )


def generate_configs(template_path, params_to_vary, out_dir=CONFIGS_TEMP_DIR):
    """Generate one merged config per point in the Cartesian product of
    ``params_to_vary`` (``dict[dotted_key -> list of values]``). Returns the
    list of written file paths."""
    with open(template_path, "r") as f:
        template = yaml.safe_load(f)

    _validate(template, params_to_vary)

    os.makedirs(out_dir, exist_ok=True)
    template_stem = os.path.splitext(os.path.basename(template_path))[0]

    keys = list(params_to_vary)
    written = []
    for combo in itertools.product(*(params_to_vary[k] for k in keys)):
        config = copy.deepcopy(template)
        fragments = []
        for dotted, value in zip(keys, combo):
            _set_dotted(config, dotted, value)
            fragments.append(_filename_fragment(dotted, value))
        out_path = os.path.join(out_dir, f"{template_stem}_{'_'.join(fragments)}.yaml")
        if os.path.exists(out_path):
            print(f"Warning: overwriting existing config {out_path}")
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        written.append(out_path)

    return written
