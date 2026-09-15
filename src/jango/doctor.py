"""Non-invasive Jango runtime readiness checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

from jango.paths import repository_root, resolve_output_root
from jango.runtime.rosetta import RosettaPreflightError, validate_rosetta_runtime_from_env


@dataclass(frozen=True)
class DoctorCheck:
    level: str
    name: str
    detail: str
    required: bool = False


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def _env_path(env: Mapping[str, str], name: str) -> Path | None:
    value = env.get(name)
    return Path(value).expanduser() if value else None


def _check(ok: bool, name: str, detail: str, *, required: bool = False, warn_detail: str | None = None) -> DoctorCheck:
    if ok:
        return DoctorCheck("PASS", name, detail, required)
    if required:
        return DoctorCheck("FAIL", name, warn_detail or detail, required)
    return DoctorCheck("WARN", name, warn_detail or detail, required)


def _nearest_existing(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _is_writable_target(path: Path) -> bool:
    target = path if path.exists() else _nearest_existing(path)
    return target.exists() and os.access(target, os.W_OK)


def _hardcoded_data_roots(text: str) -> list[str]:
    """Return hard-coded /data/<user> roots that should not live in configs."""

    allowed_roots = {"shared", "$USER", "${USER}"}
    hits: set[str] = set()
    for match in re.finditer(r"(?<![A-Za-z0-9._$}~-])/data/([^/\s\'\"]+)", text):
        root = match.group(1)
        if root in allowed_roots or "$" in root:
            continue
        hits.add(f"/data/{root}")
    return sorted(hits)


def _contains_placeholder(value: str | None) -> bool:
    if not value:
        return False
    return "<" in value or ">" in value or "${" in value and ":?" in value


def _run_probe(argv: Sequence[str], *, timeout: int = 20, quick: bool = False) -> tuple[bool, str]:
    if quick:
        return True, "probe skipped by --quick"
    try:
        completed = subprocess.run(
            list(argv),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as exc:
        return False, str(exc)
    if completed.returncode == 0:
        first = (completed.stdout or completed.stderr or "ok").strip().splitlines()
        return True, first[0][:160] if first else "ok"
    first = (completed.stderr or completed.stdout or "no output").strip().splitlines()
    return False, first[0][:160] if first else "no output"


def _check_file(env: Mapping[str, str], name: str, *, label: str | None = None, required: bool = False) -> DoctorCheck:
    path = _env_path(env, name)
    title = label or name
    if path is None:
        return _check(False, title, f"{name} is set", required=required, warn_detail=f"{name} is not set")
    return _check(path.is_file(), title, str(path), required=required, warn_detail=f"not a file: {path}")


def _check_dir(env: Mapping[str, str], name: str, *, label: str | None = None, required: bool = False) -> DoctorCheck:
    path = _env_path(env, name)
    title = label or name
    if path is None:
        return _check(False, title, f"{name} is set", required=required, warn_detail=f"{name} is not set")
    return _check(path.is_dir(), title, str(path), required=required, warn_detail=f"not a directory: {path}")


def _repo_root(env: Mapping[str, str]) -> Path:
    return Path(env.get("JANGO_SOURCE") or repository_root()).expanduser().resolve()


def check_active_python(env: Mapping[str, str]) -> list[DoctorCheck]:
    checks = [DoctorCheck("PASS", "active Python", sys.executable)]
    env_name = env.get("CONDA_DEFAULT_ENV")
    if env_name == "jango" or "/envs/jango" in sys.executable:
        checks.append(DoctorCheck("PASS", "active Conda environment", env_name or sys.executable))
    else:
        checks.append(DoctorCheck("WARN", "active Conda environment", f"expected jango, got {env_name or 'unknown'}"))
    return checks


def check_imports(env: Mapping[str, str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    source = _repo_root(env)
    for name in ["jango", "nbia"]:
        try:
            module = importlib.import_module(name)
            module_path = Path(module.__file__ or "").resolve()
        except Exception as exc:  # noqa: BLE001 - doctor reports import errors.
            checks.append(DoctorCheck("FAIL", f"import {name}", str(exc), True))
            continue
        checks.append(DoctorCheck("PASS", f"import {name}", str(module_path), True))
        try:
            module_path.relative_to(source)
            checks.append(DoctorCheck("PASS", f"{name} import path", f"under {source}"))
        except ValueError:
            checks.append(DoctorCheck("WARN", f"{name} import path", f"not under JANGO_SOURCE {source}: {module_path}"))
    return checks


def check_output_root(env: Mapping[str, str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    value = env.get("JANGO_TEST_RUN_ROOT") or env.get("JANGO_RUN_ROOT")
    if not value:
        return [DoctorCheck("WARN", "output root", "JANGO_TEST_RUN_ROOT/JANGO_RUN_ROOT is not set")]
    path = Path(value)
    try:
        resolved = resolve_output_root(path, repo_root=_repo_root(env))
    except ValueError as exc:
        return [DoctorCheck("FAIL", "output root safety", str(exc), True)]
    checks.append(DoctorCheck("PASS", "output root safety", str(resolved), True))
    checks.append(_check(_is_writable_target(resolved), "output root write permission", str(resolved), required=True, warn_detail=f"not writable or no writable existing parent: {resolved}"))
    if _contains_placeholder(value):
        checks.append(DoctorCheck("FAIL", "output root placeholders", f"unresolved placeholder in {value}", True))
    else:
        checks.append(DoctorCheck("PASS", "output root placeholders", "none"))
    return checks


def check_configs(env: Mapping[str, str], config_root: Path) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    expected = [
        "environment.yml",
        "examples/paths.example.env",
        "examples/runtime.<SLURM_NODE>.example.env",
        "test/paths.env",
        "test/esmfold2.env",
        "test/opendde.env",
        "production/paths.env",
        "production/runtime.<SLURM_NODE>.common.env",
        "production/esmfold2.env",
        "production/boltz2.env",
        "production/opendde.env",
    ]
    for rel in expected:
        path = config_root / rel
        checks.append(_check(path.is_file(), f"config {rel}", str(path), required=False, warn_detail="missing"))
    active_paths = [env.get("JANGO_PATHS_CONFIG"), env.get("JANGO_RUNTIME_CONFIG")]
    for value in [v for v in active_paths if v]:
        path = Path(value)
        if not path.is_absolute():
            path = repository_root() / path
        checks.append(_check(path.is_file(), f"active config {value}", str(path), required=True, warn_detail="missing"))
        if path.is_file():
            text = path.read_text(errors="ignore")
            if "<" in text or ">" in text:
                checks.append(DoctorCheck("FAIL", f"active config placeholders {value}", "contains angle-bracket placeholder", True))
            else:
                checks.append(DoctorCheck("PASS", f"active config placeholders {value}", "none"))
    personal_hits = []
    for rel in [
        "test/paths.env",
        "test/esmfold2.env",
        "production/paths.env",
        "production/esmfold2.env",
        "production/boltz2.env",
        "production/opendde.env",
    ]:
        path = config_root / rel
        if not path.is_file():
            continue
        hits = _hardcoded_data_roots(path.read_text(errors="ignore"))
        if hits:
            personal_hits.append(f"{rel}: {', '.join(hits)}")
    checks.append(_check(not personal_hits, "config personal data paths", "none", required=False, warn_detail="; ".join(personal_hits)))
    legacy = [config_root / "test_esm_5_paths.env", config_root / "test_esm_5_runtime.env"]
    if all(path.is_file() for path in legacy):
        checks.append(DoctorCheck("WARN", "legacy test config names", "compatibility shims are present; canonical files are under configs/test/"))
    for rel in ["paths.example.env", "runtime.<SLURM_NODE>.example.env"]:
        if (config_root / rel).exists():
            checks.append(DoctorCheck("WARN", f"legacy root config {rel}", "move examples under configs/examples/"))
    return checks


def check_repo_leakage(env: Mapping[str, str]) -> list[DoctorCheck]:
    source = _repo_root(env)
    matches: list[str] = []
    for name in ["work", "staging", "results", "logs", "cache", "outputs", "experiments"]:
        if (source / name).exists():
            matches.append(name)
    matches.extend(path.name for path in source.glob("*.pdb"))
    return [_check(not matches, "repository-root generated leakage", "none", required=True, warn_detail=", ".join(sorted(matches)))]


def check_proteinmpnn(env: Mapping[str, str], *, required: bool, quick: bool) -> list[DoctorCheck]:
    checks = [
        _check_dir(env, "PROTEINMPNN_HOME", required=required),
        _check_file(env, "PROTEINMPNN_PYTHON", required=required),
        _check_file(env, "PROTEINMPNN_RUNNER", required=required),
    ]
    python = env.get("PROTEINMPNN_PYTHON")
    runner = env.get("PROTEINMPNN_RUNNER")
    if python and runner and Path(python).is_file() and Path(runner).is_file():
        ok, detail = _run_probe([python, runner, "--help"], timeout=20, quick=quick)
        checks.append(_check(ok, "ProteinMPNN --help", detail, required=required, warn_detail=detail))
    return checks


def check_esmfold2(env: Mapping[str, str], *, required: bool, quick: bool) -> list[DoctorCheck]:
    source = _repo_root(env)
    runner = source / "scripts" / "nbia" / "run_esmfold2_monomer.py"
    checks = [
        _check_file(env, "ESMFOLD2_PYTHON", required=required),
        _check_dir(env, "ESM_ROOT", required=required),
        _check(runner.is_file(), "ESMFold2 runner", str(runner), required=required, warn_detail=f"missing: {runner}"),
    ]
    if env.get("ESMFOLD2_CACHE"):
        checks.append(_check_dir(env, "ESMFOLD2_CACHE", required=required))
    if env.get("ESMC_MODEL_PATH"):
        model = Path(env["ESMC_MODEL_PATH"]).expanduser()
        checks.append(_check(model.exists(), "ESMC model path", str(model), required=required, warn_detail=f"missing: {model}"))
    python = env.get("ESMFOLD2_PYTHON")
    if python and Path(python).is_file() and runner.is_file():
        ok, detail = _run_probe([python, str(runner), "--help"], timeout=20, quick=quick)
        checks.append(_check(ok, "ESMFold2 runner --help", detail, required=required, warn_detail=detail))
    if required and _truthy(env.get("MSA_REQUIRE")):
        if quick:
            checks.append(DoctorCheck("PASS", "ESMFold2 MSA API", "probe skipped by --quick", True))
        elif python and env.get("ESM_ROOT") and Path(python).is_file():
            code = (
                "import sys; "
                f"sys.path.insert(0, {str(Path(env['ESM_ROOT']))!r}); "
                "from esm.utils.msa import MSA; "
                "from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput; "
                "from esm.models.esmfold2.processor import ESMFold2InputBuilder; "
                "print('esmfold2_msa_api_ok')"
            )
            ok, detail = _run_probe([python, "-c", code], timeout=30, quick=False)
            checks.append(_check(ok, "ESMFold2 MSA API", detail, required=True, warn_detail=detail))
    return checks


def check_boltz2(env: Mapping[str, str], *, required: bool, quick: bool) -> list[DoctorCheck]:
    checks = []
    executable = _env_path(env, "BOLTZ_EXECUTABLE") or _env_path(env, "BOLTZ")
    checks.append(_check(executable is not None and executable.exists(), "BOLTZ_EXECUTABLE", str(executable) if executable else "not set", required=required, warn_detail=f"missing: {executable or 'BOLTZ_EXECUTABLE'}"))
    if env.get("BOLTZ_PYTHON"):
        checks.append(_check_file(env, "BOLTZ_PYTHON", required=required))
    if env.get("BOLTZ_MODEL_ROOT"):
        checks.append(_check_dir(env, "BOLTZ_MODEL_ROOT", required=required))
    if executable and executable.exists():
        ok, detail = _run_probe([str(executable), "--help"], timeout=20, quick=quick)
        checks.append(_check(ok, "Boltz2 --help", detail, required=required, warn_detail=detail))
    return checks


def check_opendde(env: Mapping[str, str], *, required: bool, quick: bool) -> list[DoctorCheck]:
    executable = _env_path(env, "OPENDDE_EXECUTABLE") or _env_path(env, "OPENDDE_BIN")
    checks = [
        _check_file(env, "OPENDDE_PYTHON", required=required),
        _check(executable is not None and executable.is_file(), "OPENDDE_EXECUTABLE", str(executable) if executable else "not set", required=required, warn_detail=f"missing: {executable or 'OPENDDE_EXECUTABLE'}"),
        _check_dir(env, "OPENDDE_ROOT_DIR", required=required) if env.get("OPENDDE_ROOT_DIR") else _check_dir(env, "OPENDDE_MODEL_ROOT", label="OPENDDE_ROOT_DIR", required=required),
    ]
    if env.get("OPENDDE_CHECKPOINT"):
        checks.append(_check_file(env, "OPENDDE_CHECKPOINT", required=required))
    if executable and executable.is_file():
        ok, detail = _run_probe([str(executable), "--help"], timeout=20, quick=quick)
        checks.append(_check(ok, "OpenDDE --help", detail, required=required, warn_detail=detail))
    return checks


def check_msa(env: Mapping[str, str]) -> list[DoctorCheck]:
    provider = env.get("MSA_PROVIDER") or "none"
    required = _truthy(env.get("MSA_REQUIRE")) or env.get("MSA_MODE") == "required"
    checks = [DoctorCheck("PASS", "MSA provider", provider)]
    if provider == "abforge_get_or_build":
        checks.append(_check_file(env, "MSA_SCRIPT", required=required))
        checks.append(_check_dir(env, "MSA_CACHE_ROOT", required=required))
    elif provider in {"precomputed", "mmseqs2"}:
        checks.append(_check_dir(env, "MSA_CACHE_ROOT", required=required))
    if _truthy(env.get("MSA_BUILD_IF_MISSING")):
        search = env.get("MSA_SEARCH_BINARY") or env.get("COLABFOLD_SEARCH") or env.get("MMSEQS")
        database = env.get("MSA_DATABASE") or env.get("COLABFOLD_DB") or env.get("MMSEQS_DB")
        checks.append(_check(bool(search), "MSA search executable", search or "not set", required=required, warn_detail="set MSA_SEARCH_BINARY/COLABFOLD_SEARCH/MMSEQS"))
        if search:
            path = Path(search).expanduser()
            checks.append(_check(path.exists() or shutil.which(search) is not None, "MSA search executable path", search, required=required, warn_detail=f"missing: {search}"))
        checks.append(_check(bool(database), "MSA database", database or "not set", required=required, warn_detail="set MSA_DATABASE/COLABFOLD_DB/MMSEQS_DB"))
        if database:
            checks.append(_check(Path(database).expanduser().exists(), "MSA database path", database, required=required, warn_detail=f"missing: {database}"))
    else:
        checks.append(DoctorCheck("WARN", "MSA search/database", "MSA_BUILD_IF_MISSING=false; doctor verifies cache/helper only"))
    return checks


def check_rosetta(env: Mapping[str, str], *, quick: bool) -> list[DoctorCheck]:
    if not env.get("ROSETTA_RUNTIME"):
        return [DoctorCheck("WARN", "Rosetta runtime", "ROSETTA_RUNTIME is not set")]
    if quick:
        return [DoctorCheck("PASS", "Rosetta runtime", "preflight skipped by --quick", True)]
    try:
        result = validate_rosetta_runtime_from_env(env, timeout_seconds=120)
    except RosettaPreflightError as exc:
        return [DoctorCheck("FAIL", "Rosetta runtime", str(exc), True)]
    if result is None:
        return [DoctorCheck("WARN", "Rosetta runtime", "not configured")]
    checked = ", ".join(result.executables)
    return [DoctorCheck("PASS", "Rosetta runtime", f"{result.method}; checked {checked}", True)]


def check_dockq(env: Mapping[str, str], *, quick: bool) -> list[DoctorCheck]:
    provider = env.get("DOCKQ_PROVIDER")
    if provider == "dockq_rs_python":
        python = env.get("DOCKQ_PYTHON")
        checks = [_check(bool(python) and Path(python).is_file(), "DockQ-RS Python", python or "not set", required=True, warn_detail=f"missing: {python or 'DOCKQ_PYTHON'}")]
        if python and Path(python).is_file():
            ok, detail = _run_probe([python, "-c", "from dockq_rs import score_pairs; print('score_pairs ok')"], timeout=20, quick=quick)
            checks.append(_check(ok, "DockQ-RS score_pairs", detail, required=True, warn_detail=detail))
        return checks
    if env.get("DOCKQ_BIN"):
        return [_check_file(env, "DOCKQ_BIN", label="DockQ executable", required=True)]
    return [DoctorCheck("WARN", "DockQ", "DOCKQ_PROVIDER/DOCKQ_BIN is not set")]


def check_anarci(env: Mapping[str, str], *, quick: bool) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    python = env.get("ANARCI_PYTHON")
    binary = env.get("ANARCI_BIN")
    if python:
        checks.append(_check_file(env, "ANARCI_PYTHON", required=True))
        if Path(python).is_file():
            ok, detail = _run_probe([python, "-c", "import anarci; print('anarci import ok')"], timeout=20, quick=quick)
            checks.append(_check(ok, "ANARCI Python import", detail, required=True, warn_detail=detail))
    if binary:
        checks.append(_check_file(env, "ANARCI_BIN", required=False))
        if Path(binary).is_file():
            ok, detail = _run_probe([binary, "--help"], timeout=20, quick=quick)
            checks.append(_check(ok, "ANARCI CLI --help", detail, required=False, warn_detail=detail))
    if not checks:
        checks.append(DoctorCheck("WARN", "ANARCI", "ANARCI_PYTHON/ANARCI_BIN is not set; precomputed numbering table required for mutation localization"))
    return checks


def check_slurm(env: Mapping[str, str]) -> list[DoctorCheck]:
    if env.get("EXECUTION_MODE") != "slurm":
        return [DoctorCheck("WARN", "Slurm", f"EXECUTION_MODE={env.get('EXECUTION_MODE', 'unset')}; Slurm not required")]
    checks: list[DoctorCheck] = []
    for cmd in ["sbatch", "squeue", "sacct"]:
        found = shutil.which(cmd)
        checks.append(_check(found is not None, f"Slurm command {cmd}", found or "not found", required=True, warn_detail="not found on PATH"))
    for name in ["SLURM_PARTITION", "SLURM_GPUS", "SLURM_CPUS_PER_TASK", "SLURM_MEM", "SLURM_TIME", "SLURM_MAX_CONCURRENT", "FOLD_WORKERS_PER_GPU"]:
        checks.append(_check(bool(env.get(name)), f"Slurm resource {name}", env.get(name, "not set"), required=True, warn_detail="not set"))
    return checks


def collect_checks(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> list[DoctorCheck]:
    values = os.environ if env is None else env
    backend = args.backend or values.get("FOLDING_BACKEND")
    config_root = args.config_root or repository_root() / "configs"
    checks: list[DoctorCheck] = []
    checks.extend(check_active_python(values))
    checks.extend(check_imports(values))
    checks.extend(check_configs(values, config_root))
    checks.extend(check_output_root(values))
    checks.extend(check_repo_leakage(values))
    checks.extend(check_proteinmpnn(values, required=backend in {"esmfold2", "boltz2", "opendde"}, quick=args.quick))
    if backend in {None, "esmfold2"}:
        checks.extend(check_esmfold2(values, required=backend == "esmfold2", quick=args.quick))
    if backend in {None, "boltz2"}:
        checks.extend(check_boltz2(values, required=backend == "boltz2", quick=args.quick))
    if backend in {None, "opendde"}:
        checks.extend(check_opendde(values, required=backend == "opendde", quick=args.quick))
    checks.extend(check_msa(values))
    checks.extend(check_rosetta(values, quick=args.quick))
    checks.extend(check_dockq(values, quick=args.quick))
    checks.extend(check_anarci(values, quick=args.quick))
    checks.extend(check_slurm(values))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jango doctor", description="Non-invasive runtime and configuration readiness checks.")
    parser.add_argument("--backend", choices=["esmfold2", "boltz2", "opendde"], default=None, help="Override FOLDING_BACKEND for relevance/exit-code decisions.")
    parser.add_argument("--config-root", type=Path, default=None, help="Config directory to inspect; defaults to repository configs/.")
    parser.add_argument("--quick", action="store_true", help="Skip external --help/import/preflight subprocess probes.")
    parser.add_argument("--verbose", action="store_true", help="Accepted for compatibility; output is already grouped.")
    return parser


def print_grouped(checks: list[DoctorCheck]) -> None:
    for level in ["PASS", "WARN", "FAIL"]:
        print(level)
        group = [check for check in checks if check.level == level]
        if not group:
            print("  - none")
            continue
        for check in group:
            suffix = " [required]" if check.required and check.level == "FAIL" else ""
            print(f"  - {check.name}: {check.detail}{suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv or []))
    checks = collect_checks(args)
    print_grouped(checks)
    return 1 if any(check.level == "FAIL" and check.required for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
