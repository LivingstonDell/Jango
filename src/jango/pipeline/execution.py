"""Shared command execution helpers for DELPHI pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


CommandPart = str | Path


@dataclass(frozen=True)
class PipelineCommand:
    """One external command in a pipeline stage."""

    name: str
    argv: tuple[CommandPart, ...]
    cwd: Path | None = None

    def display(self) -> str:
        return " ".join(str(part) for part in self.argv)


def python_module_command(module: str, *parts: CommandPart, name: str | None = None) -> PipelineCommand:
    return PipelineCommand(name=name or module, argv=(sys.executable, "-m", module, *parts))


def nbia_command(*parts: CommandPart, name: str | None = None) -> PipelineCommand:
    return python_module_command("nbia.cli", *parts, name=name or f"nbia {' '.join(str(p) for p in parts[:1])}")


def build_env(
    *,
    atlas_root: Path | None = None,
    extra: Mapping[str, str | Path | None] | None = None,
) -> dict[str, str]:
    """Build a subprocess environment with the Atlas source tree on PYTHONPATH."""

    env = os.environ.copy()
    pythonpath: list[str] = []
    if atlas_root is not None:
        atlas_root = atlas_root.expanduser().resolve()
        env["NBIA_ROOT"] = str(atlas_root)
        atlas_src = atlas_root / "src"
        if atlas_src.exists():
            pythonpath.append(str(atlas_src))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if extra:
        for key, value in extra.items():
            if value is not None:
                env[key] = str(value)
    return env


class CommandRunner:
    """Execute or print pipeline commands."""

    def __init__(self, *, env: Mapping[str, str] | None = None, plan_only: bool = False) -> None:
        self.env = dict(env) if env is not None else None
        self.plan_only = plan_only

    def run(self, command: PipelineCommand, *, cwd: Path | None = None) -> None:
        actual_cwd = cwd if cwd is not None else command.cwd
        if self.plan_only:
            location = f"  cwd: {actual_cwd}" if actual_cwd is not None else ""
            print(f"[plan] {command.name}: {command.display()}{location}")
            return

        print("\nRunning:")
        print(command.display())
        result = subprocess.run(
            [str(part) for part in command.argv],
            env=self.env,
            cwd=actual_cwd,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    def run_many(self, commands: Sequence[PipelineCommand], *, cwd: Path | None = None) -> None:
        for command in commands:
            self.run(command, cwd=cwd)


def ensure_file(path: Path, label: str | None = None) -> None:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Required {label or 'file'} not found: {path}")


def ensure_dir(path: Path, label: str | None = None) -> None:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Required {label or 'directory'} not found: {path}")


def ensure_writable_dir(path: Path, label: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    try:
        probe.write_text("ok\n")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(f"{label or 'directory'} is not writable: {path} ({exc})") from exc
    return path


def resolve_path(path: Path, root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).expanduser().resolve()
