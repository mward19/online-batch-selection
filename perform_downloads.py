"""
To be used before a run on a supercomputer compute node, to download necessary
data on a login node. Right now, just downloads the CLIP model
"""


import os
import yaml
import argparse

def download_clip(method_config_path):
    with open(method_config_path) as f:
        config = yaml.safe_load(f)
    try:
        arch = config["clip"]["clip_architecture"]
    except (KeyError, TypeError):
        print(f"CLIP architecture not listed in config. Not downloading.")
        return

    filename = arch.replace("/", "-") + ".pt"
    path = os.path.join("./models/teacher", filename)
    if os.path.exists(path):
        print(f"CLIP arch {arch} already present at {path}. Skipping download.")
        return

    print(f"Downloading CLIP arch: {arch}")
    import clip
    clip.load(arch, download_root="./models/teacher", jit=False, device="cpu")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    download_clip(args.method)