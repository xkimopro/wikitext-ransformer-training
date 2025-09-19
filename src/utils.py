import os
import random
import time
import csv
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def env_cache_setup():
    hf_home = os.getenv("HF_HOME")
    hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    print("--- Cache Environment ---")
    print(f"HF_HOME: {hf_home}")
    print(f"HF_DATASETS_CACHE: {hf_datasets_cache}")
    print("-------------------------")
    for path in [hf_home, hf_datasets_cache]:
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


class CSVLogger:
    """
    Simple CSV logger that ALWAYS writes a header row when constructed,
    so each run (training or profiling) starts with headers in the file.
    """
    def __init__(self, filepath, fieldnames):
        self.filepath = Path(filepath)
        self.fieldnames = list(fieldnames)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Always append a header for this run (even if the file exists).
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, data: dict):
        # Missing keys will be written as blanks.
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)


class Timer:
    """
    Wall-clock timer; optionally CUDA-synchronized for accurate GPU timing.
    """
    def __init__(self, sync_cuda: bool = False):
        self.sync_cuda = sync_cuda
        self.start_time = None
        self.elapsed_time = 0.0

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_time = time.monotonic() - self.start_time

    def __float__(self):
        return self.elapsed_time
