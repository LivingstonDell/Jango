"""Rosetta runtime configuration for DELPHI orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Literal, Mapping, Sequence

from nbia.config import RosettaRuntimeConfig, RosettaRuntimeKind


RosettaScoreState = Literal["native_relaxed", "decoy_relaxed"]

REQUIRED_ROSETTA_EXECUTABLES = (
    "InterfaceAnalyzer.default.linuxgccrelease",
    "rosetta_scripts.default.linuxgccrelease",
)
ROSETTA_CONTAINER_BIN_DIR = "/usr/local/bin"


def ensure_matching_rosetta_states(native_state: str, decoy_state: str) -> None:
    """Refuse native/decoy comparisons that are not relaxed-vs-relaxed."""

    if native_state != "native_relaxed":
        raise ValueError("native Rosetta score state must be native_relaxed")
    if decoy_state != "decoy_relaxed":
        raise ValueError("decoy Rosetta score state must be decoy_relaxed")


class RosettaPreflightError(RuntimeError):
    """Raised when a configured Rosetta runtime fails a non-scientific preflight."""


@dataclass(frozen=True)
class RosettaPreflightResult:
    """Summary of a successful runtime preflight."""

    kind: str
    method: str
    executables: tuple[str, ...]
    image: str | None = None
    bin_dir: Path | None = None


def _env_value(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value if value else None


def _env_path(env: Mapping[str, str], name: str) -> Path | None:
    value = _env_value(env, name)
    return Path(value).expanduser() if value else None


def rosetta_runtime_config_from_env(env: Mapping[str, str] | None = None) -> RosettaRuntimeConfig | None:
    """Build Rosetta runtime config from environment variables, if configured."""

    values = os.environ if env is None else env
    kind = _env_value(values, "ROSETTA_RUNTIME")
    if kind is None:
        return None
    if kind not in {"docker", "apptainer", "local", "executable"}:
        raise RosettaPreflightError(f"Unsupported ROSETTA_RUNTIME: {kind}")
    return RosettaRuntimeConfig(
        kind=kind,  # type: ignore[arg-type]
        image=_env_value(values, "ROSETTA_IMAGE"),
        bin_dir=_env_path(values, "ROSETTA_BIN_DIR"),
        apptainer_cache_dir=_env_path(values, "APPTAINER_CACHEDIR"),
    )


def validate_rosetta_runtime_from_env(
    env: Mapping[str, str] | None = None,
    *,
    timeout_seconds: int = 120,
) -> RosettaPreflightResult | None:
    """Validate the configured runtime from env vars, returning None when absent."""

    values = os.environ if env is None else env
    config = rosetta_runtime_config_from_env(values)
    if config is None:
        return None
    return validate_rosetta_runtime(
        config,
        apptainer_bin=_env_value(values, "APPTAINER_BIN"),
        env=values,
        timeout_seconds=timeout_seconds,
    )


def validate_rosetta_runtime(
    config: RosettaRuntimeConfig,
    *,
    executables: Sequence[str] = REQUIRED_ROSETTA_EXECUTABLES,
    apptainer_bin: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
) -> RosettaPreflightResult:
    """Run a non-scientific validation for the configured Rosetta runtime."""

    config.validate()
    kind = "local" if config.kind == "executable" else config.kind
    required = tuple(executables)
    if kind == "apptainer":
        return _validate_apptainer_rosetta(
            config,
            executables=required,
            apptainer_bin=apptainer_bin,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    if kind == "local":
        return _validate_local_rosetta(config, executables=required)
    if kind == "docker":
        raise RosettaPreflightError("Docker Rosetta runtime doctor validation is not implemented; use command-level validation.")
    raise RosettaPreflightError(f"Unsupported Rosetta runtime: {config.kind}")


def _validate_local_rosetta(config: RosettaRuntimeConfig, *, executables: tuple[str, ...]) -> RosettaPreflightResult:
    bin_dir = config.bin_dir
    if bin_dir is None or not bin_dir.is_dir():
        raise RosettaPreflightError(f"Local Rosetta bin directory is unavailable: {bin_dir}")
    missing = [name for name in executables if not (bin_dir / name).is_file()]
    if missing:
        raise RosettaPreflightError(f"Missing local Rosetta executables in {bin_dir}: {', '.join(missing)}")
    return RosettaPreflightResult(
        kind="local",
        method="local executable check",
        executables=executables,
        bin_dir=bin_dir,
    )


def _resolve_apptainer_bin(apptainer_bin: str | None) -> str:
    configured = apptainer_bin or shutil.which("apptainer")
    if not configured:
        raise RosettaPreflightError("Apptainer Rosetta runtime requested, but apptainer is not available on PATH.")
    path = Path(configured).expanduser()
    if path.is_absolute() and not path.exists():
        raise RosettaPreflightError(f"Configured Apptainer executable does not exist: {path}")
    return str(path if path.is_absolute() else configured)


def _container_executable_test(executables: tuple[str, ...]) -> str:
    tests = []
    for name in executables:
        path = f"{ROSETTA_CONTAINER_BIN_DIR}/{name}"
        tests.append(f"test -x {shlex.quote(path)}")
    return " && ".join(tests)


def _validate_apptainer_rosetta(
    config: RosettaRuntimeConfig,
    *,
    executables: tuple[str, ...],
    apptainer_bin: str | None,
    env: Mapping[str, str] | None,
    timeout_seconds: int,
) -> RosettaPreflightResult:
    image = config.image
    if not image:
        raise RosettaPreflightError("Apptainer Rosetta runtime requires ROSETTA_IMAGE/--rosetta-image.")
    if image.startswith("docker://"):
        method = "apptainer exec docker URI executable check"
    else:
        image_path = Path(image).expanduser()
        if not image_path.is_file():
            raise RosettaPreflightError(f"Local Apptainer image is unavailable: {image_path}")
        image = str(image_path)
        method = "apptainer exec local image executable check"

    command = [_resolve_apptainer_bin(apptainer_bin), "exec", image, "sh", "-lc", _container_executable_test(executables)]
    run_env = os.environ.copy()
    if env is not None:
        run_env.update(env)
    if config.apptainer_cache_dir is not None:
        run_env["APPTAINER_CACHEDIR"] = str(config.apptainer_cache_dir)

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=run_env,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RosettaPreflightError(f"Apptainer Rosetta executable check timed out after {timeout_seconds}s") from exc
    except OSError as exc:
        raise RosettaPreflightError(f"Unable to launch Apptainer executable check: {exc}") from exc

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "no output").strip().splitlines()
        message = details[0] if details else "no output"
        raise RosettaPreflightError(f"Apptainer Rosetta executable check failed: {message}")
    return RosettaPreflightResult(
        kind="apptainer",
        method=method,
        executables=executables,
        image=image,
    )


