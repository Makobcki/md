from diffusion.perf.triton_compat import _move_python_include_before_cuda


def test_move_python_include_before_cuda() -> None:
    src = """#include "cuda.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
"""

    patched = _move_python_include_before_cuda(src)

    assert patched.index("#include <Python.h>") < patched.index('#include "cuda.h"')
    assert patched.count("#define PY_SSIZE_T_CLEAN") == 1
    assert patched.count("#include <Python.h>") == 1


def test_keep_source_when_python_include_is_already_first() -> None:
    src = """#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cuda.h"
"""

    assert _move_python_include_before_cuda(src) == src
