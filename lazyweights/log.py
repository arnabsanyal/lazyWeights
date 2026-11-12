"""JSONL result logging, config hashing, seeding and environment capture.

Python 3.6-compatible (runs on the Jetson Nano).
"""
import hashlib
import json
import os
import random
import socket
import subprocess
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def git_commit():
    try:
        out = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "-C", REPO_ROOT, "diff", "--quiet", "HEAD", "--", "lazyweights", "scripts"],
                                stderr=subprocess.DEVNULL)
        return out.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def seed_all(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class JsonlLogger(object):
    """Appends rows to results/<experiment_id>.jsonl with the provenance fields."""

    def __init__(self, experiment_id, cfg=None, seed=0, machine=None, path=None):
        self.experiment_id = experiment_id
        self.cfg = cfg or {}
        self.seed = seed
        self.machine = machine or socket.gethostname()
        self.path = path or os.path.join(RESULTS_DIR, experiment_id + ".jsonl")
        self.base = {
            "experiment_id": experiment_id,
            "config_hash": config_hash(self.cfg),
            "seed": seed,
            "git_commit": git_commit(),
            "machine": self.machine,
        }
        d = os.path.dirname(self.path)
        if d and not os.path.isdir(d):
            os.makedirs(d)

    def log(self, row, **extra):
        rec = dict(self.base)
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec.update(extra)
        rec.update(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=_default) + "\n")
        return rec


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)
