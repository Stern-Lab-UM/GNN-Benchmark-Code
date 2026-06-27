#!/usr/bin/env python
"""Check whether the curated DCG/GNN code can import in this environment.

This script is intentionally lightweight: it does not train a model, load a
checkpoint, or require manuscript data. It verifies Python/package availability
and imports the curated source snapshots from this repository.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("click", "click"),
    ("tqdm", "tqdm"),
    ("hdf5storage", "hdf5storage"),
]

MPNN_IMPORTS = [
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
]

PPGN_IMPORTS = [
    ("torch", "torch"),
]


def _module_version(module: object) -> str:
    """
    Return a displayable version string for an imported module.

    Args:
        module: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    return str(getattr(module, "__version__", "unknown"))


def _clear_modules(prefix: str) -> None:
    """
    Remove cached modules with the requested prefix from sys.modules.

    Args:
        prefix: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


@contextmanager
def _prepended_path(path: Path):
    """
    Temporarily prepend a path to sys.path during a check.

    Args:
        path: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def check_imports(imports: list[tuple[str, str]]) -> bool:
    """
    Verify that required Python imports are available.

    Args:
        imports: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    ok = True
    for module_name, package_name in imports:
        try:
            module = importlib.import_module(module_name)
            print(f"[OK] import {module_name:<18} version={_module_version(module)}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            ok = False
            print(f"[FAIL] import {module_name:<18} package={package_name} error={exc}")
    return ok


def check_mpnn_source() -> bool:
    """
    Verify that the MPNN source tree can be imported.

    Returns:
        Computed value used by the caller.
    """
    src = REPO_ROOT / "models" / "mpnn"
    if not src.exists():
        print(f"[FAIL] missing MPNN source directory: {src}")
        return False

    ok = True
    with _prepended_path(src):
        for prefix in ("dataset", "trainer_final", "predict_final", "models"):
            _clear_modules(prefix)
        for module_name in ("dataset", "models", "trainer_final", "predict_final"):
            try:
                module = importlib.import_module(module_name)
                print(f"[OK] import MPNN source module {module_name} from {getattr(module, '__file__', 'built-in')}")
                if module_name == "models":
                    for attr in ("GraphSAGE", "GIN", "PNA", "EdgeRegressor"):
                        if not hasattr(module, attr):
                            ok = False
                            print(f"[FAIL] models missing expected attribute {attr}")
            except Exception as exc:  # pragma: no cover - diagnostic script
                ok = False
                print(f"[FAIL] import MPNN source module {module_name}: {exc}")
    return ok


def check_ppgn_snapshot(snapshot: str) -> bool:
    """
    Verify that a PPGN package snapshot can be imported.

    Args:
        snapshot: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    src = REPO_ROOT / "models" / "ppgn" / snapshot
    if not src.exists():
        print(f"[FAIL] missing PPGN source directory: {src}")
        return False

    ok = True
    with _prepended_path(src):
        _clear_modules("dcg")
        for module_name in (
            "dcg.base_model",
            "dcg.cell_loader",
            "dcg.configuration",
            "dcg.file_reader",
            "dcg.main",
            "dcg.modules",
            "dcg.recorder",
            "dcg.simulator",
            "dcg.trainer",
        ):
            try:
                module = importlib.import_module(module_name)
                print(f"[OK] import {snapshot} module {module_name} from {getattr(module, '__file__', 'built-in')}")
            except Exception as exc:  # pragma: no cover - diagnostic script
                ok = False
                print(f"[FAIL] import {snapshot} module {module_name}: {exc}")
        _clear_modules("dcg")
    return ok


def check_ppgn_source() -> bool:
    """
    Verify that the PPGN source snapshots can be imported.

    Returns:
        Computed value used by the caller.
    """
    ok = True
    for snapshot in ("train_dcg", "predict_dcg", "gl_tail_fixed_pkg"):
        ok = check_ppgn_snapshot(snapshot) and ok
    return ok


def print_header() -> None:
    """
    Print a section header for the installation checker.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    print("DCG/GNN install checker")
    print(f"repo: {REPO_ROOT}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    if sys.version_info < (3, 10):
        print("[WARN] Python 3.10+ is recommended for these curated snapshots.")


def main() -> int:
    """
    Parse command-line arguments and run this script entry point.

    Returns:
        Computed value used by the caller.
    """
    parser = argparse.ArgumentParser(
        description="Check imports for the curated DCG/GNN publication code."
    )
    parser.add_argument(
        "--component",
        choices=("all", "mpnn", "ppgn"),
        default="all",
        help="Which code family to check.",
    )
    parser.add_argument(
        "--packages-only",
        action="store_true",
        help="Only check third-party imports; skip repository source imports.",
    )
    args = parser.parse_args()

    print_header()
    print("")

    imports = list(BASE_IMPORTS)
    if args.component in ("all", "mpnn"):
        imports.extend(MPNN_IMPORTS)
    if args.component in ("all", "ppgn"):
        imports.extend(PPGN_IMPORTS)

    seen = set()
    unique_imports = []
    for item in imports:
        if item[0] not in seen:
            unique_imports.append(item)
            seen.add(item[0])

    ok = check_imports(unique_imports)

    if not args.packages_only:
        if args.component in ("all", "mpnn"):
            print("\nChecking MPNN source snapshot")
            ok = check_mpnn_source() and ok
        if args.component in ("all", "ppgn"):
            print("\nChecking PPGN source snapshots")
            ok = check_ppgn_source() and ok

    print("")
    if ok:
        print("[OK] install check passed")
        return 0

    print("[FAIL] install check failed")
    print("Install missing packages, then rerun this checker.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


