from __future__ import annotations

import sys
from functools import wraps
from typing import Any


_CUDA_INCLUDE = '#include "cuda.h"'
_PYTHON_INCLUDE = "#include <Python.h>"
_PY_SSIZE_DEFINE = "#define PY_SSIZE_T_CLEAN"
_PATCH_MARKER = "_md_python314_cuda_include_patch"


def _move_python_include_before_cuda(src: str) -> str:
    cuda_idx = src.find(_CUDA_INCLUDE)
    python_idx = src.find(_PYTHON_INCLUDE)
    if cuda_idx < 0 or python_idx < 0 or python_idx < cuda_idx:
        return src

    src = src.replace(f"{_PY_SSIZE_DEFINE}\n", "", 1)
    src = src.replace(f"{_PYTHON_INCLUDE}\n", "", 1)
    cuda_idx = src.find(_CUDA_INCLUDE)
    if cuda_idx < 0:
        return src
    python_header = f"{_PY_SSIZE_DEFINE}\n{_PYTHON_INCLUDE}\n"
    return f"{src[:cuda_idx]}{python_header}{src[cuda_idx:]}"


def patch_triton_cuda_python_include_order() -> bool:
    """Avoid Python 3.14/glibc _POSIX_C_SOURCE warnings in Triton C helpers."""
    if sys.version_info < (3, 14):
        return False

    try:
        from triton.runtime import build as triton_build
    except Exception:
        return False

    original = triton_build.compile_module_from_src
    if getattr(original, _PATCH_MARKER, False):
        return True

    @wraps(original)
    def compile_module_from_src(src: str, name: str, *args: Any, **kwargs: Any) -> Any:
        return original(_move_python_include_before_cuda(src), name, *args, **kwargs)

    setattr(compile_module_from_src, _PATCH_MARKER, True)
    triton_build.compile_module_from_src = compile_module_from_src

    try:
        from triton.backends import nvidia as nvidia_backend

        driver = getattr(nvidia_backend, "driver", None)
        if driver is not None:
            driver.compile_module_from_src = compile_module_from_src
    except Exception:
        pass

    return True
