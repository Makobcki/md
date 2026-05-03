from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Keep CPU torch tests deterministic and avoid OpenMP oversubscription/deadlocks
# in constrained CI containers. These settings only affect the test process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_ignore_collect(collection_path: Path, config: object) -> bool:
    torch_test_files = {"test_attention_masks.py", "test_collate.py", "test_conditioning_sanity.py"}
    if collection_path.name in torch_test_files and importlib.util.find_spec("torch") is None:
        return True
    return False
