import subprocess
import time
import argparse
import glob
import os

def sync(save_dirs):
    start = time.time()

    dirs = open(save_dirs).read().splitlines() if save_dirs.endswith('.txt') else [save_dirs]

    wandb_dirs = []
    for d in dirs:
        wandb_dirs.extend(glob.glob(f"{d}/wandb/offline-run-*", recursive=True))

    if wandb_dirs:
        cmd = ["wandb", "sync", *wandb_dirs, "--no-include-synced"]
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
    else:
        print("No data found to sync")

    return time.time() - start

def sync_daemon(save_dirs, interval_sec=30):
    while True:
        print(f"Syncing... ({interval_sec}s interval)")
        elapsed = sync(save_dirs)
        print(f"Sync took {elapsed:.2f}s")

        sleep_time = max(0, interval_sec - elapsed)
        end_time = time.time() + sleep_time

        # Display time remaining before next sync
        while True:
            remaining = end_time - time.time()
            if remaining <= 0:
                print(f"\rNext sync in 0.0s          ", flush=True)
                break

            print(f"\rNext sync in {remaining:.1f}s          ", end="", flush=True)
            time.sleep(0.1)

        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dirs', type=str, default='./exp/**')
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    sync_daemon(args.save_dirs, args.interval)