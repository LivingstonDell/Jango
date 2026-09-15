"""Preflight validation support for the Fett orchestration command."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Callable, Iterable, Literal, Mapping, Sequence

Severity = Literal["FAIL", "WARN", "INFO", "PASS"]
Runner = Callable[..., subprocess.CompletedProcess[str]]

PLACEHOLDER_TOKENS = ("TODO", "CHANGEME", "YOUR_PATH", "<", ">")
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".zip")
PATH_ENV_NAMES = {
    "JANGO_SOURCE",
    "JANGO_RUN_ROOT",
    "JANGO_OUTPUT_ROOT",
    "JANGO_RAW_TARBALLS",
    "PROTEINMPNN_HOME",
    "PROTEINMPNN_PYTHON",
    "PROTEINMPNN_RUNNER",
    "ESMFOLD2_PYTHON",
    "ESM_ROOT",
    "ESMFOLD2_CACHE",
    "ESMC_MODEL_PATH",
    "BOLTZ_EXECUTABLE",
    "BOLTZ_PYTHON",
    "BOLTZ_MODEL_ROOT",
    "BOLTZ_CACHE",
    "MSA_SCRIPT",
    "MSA_CACHE_ROOT",
    "DOCKQ_PYTHON",
    "ANARCI_PYTHON",
    "ANARCI_BIN",
    "APPTAINER_BIN",
    "APPTAINER_CACHEDIR",
}


@dataclass(frozen=True)
class PreflightIssue:
    severity: Severity
    code: str
    summary: str
    details: str | None = None
    remediation: str | None = None
    stage: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "severity": self.severity,
            "code": self.code,
            "summary": self.summary,
            "details": self.details,
            "remediation": self.remediation,
            "stage": self.stage,
        }


@dataclass
class PreflightReport:
    run_id: str
    issues: list[PreflightIssue] = field(default_factory=list)
    resolved_config: dict[str, str] = field(default_factory=dict)
    generated_scripts: list[Path] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(issue.severity == "FAIL" for issue in self.issues)

    def add(
        self,
        severity: Severity,
        code: str,
        summary: str,
        *,
        details: str | None = None,
        remediation: str | None = None,
        stage: str | None = None,
    ) -> None:
        self.issues.append(PreflightIssue(severity, code, summary, details, remediation, stage))

    def pass_(self, code: str, summary: str, *, details: str | None = None, stage: str | None = None) -> None:
        self.add("PASS", code, summary, details=details, stage=stage)

    def warn(self, code: str, summary: str, *, details: str | None = None, remediation: str | None = None, stage: str | None = None) -> None:
        self.add("WARN", code, summary, details=details, remediation=remediation, stage=stage)

    def fail(self, code: str, summary: str, *, details: str | None = None, remediation: str | None = None, stage: str | None = None) -> None:
        self.add("FAIL", code, summary, details=details, remediation=remediation, stage=stage)

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": "FAILED" if self.failed else "PASSED",
            "issues": [issue.as_dict() for issue in self.issues],
            "resolved_config": dict(sorted(self.resolved_config.items())),
            "generated_scripts": [str(path) for path in self.generated_scripts],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def render(self) -> str:
        lines = [f"Fett preflight: {self.run_id}", ""]
        for severity in ("PASS", "WARN", "FAIL"):
            group = [issue for issue in self.issues if issue.severity == severity]
            if not group:
                continue
            lines.append(severity)
            for issue in group:
                stage = f" [{issue.stage}]" if issue.stage else ""
                lines.append(f"  - {issue.code}{stage}: {issue.summary}")
                if issue.details:
                    for detail_line in issue.details.splitlines():
                        lines.append(f"      {detail_line}")
                if issue.remediation:
                    lines.append(f"      fix: {issue.remediation}")
            lines.append("")
        lines.append("Result: FAILED - no jobs submitted" if self.failed else "Result: PASSED")
        return "\n".join(lines)


def has_placeholder(value: object) -> bool:
    text = str(value or "")
    upper = text.upper()
    return any(token in text or token in upper for token in PLACEHOLDER_TOKENS)


def has_unresolved_expansion(value: object) -> bool:
    text = str(value or "")
    return "${" in text or "$USER" in text or re.search(r"\$[A-Za-z_][A-Za-z0-9_]*", text) is not None


def has_unmatched_braces(value: object) -> bool:
    text = str(value or "")
    return text.count("{") != text.count("}") or "}" in text or "{" in text


def has_suspicious_path(value: object) -> bool:
    text = str(value or "")
    if not text:
        return False
    if has_unresolved_expansion(text) or has_unmatched_braces(text):
        return True
    if re.search(r"(/[^/\s]+){2,}", text) and "}/" in text:
        return True
    for fragment in ("/jango/cache/apptainer", "/jango/cache/boltz"):
        if text.count(fragment) > 1:
            return True
    return False


def path_is_executable(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def shell_quote_env(value: str | Path) -> str:
    import shlex

    return shlex.quote(str(value))


def required_variables_from_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    found: dict[str, str] = {}
    for match in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:\?([^}]*)\}", text):
        found[match.group(1)] = match.group(2)
    return found


def evaluate_configs(
    *,
    paths_config: Path,
    runtime_config: Path,
    base_env: Mapping[str, str],
    runner: Runner = subprocess.run,
    timeout: int = 20,
) -> tuple[dict[str, str], list[PreflightIssue]]:
    issues: list[PreflightIssue] = []
    for label, path in (("paths config", paths_config), ("runtime config", runtime_config)):
        if not path.exists():
            issues.append(
                PreflightIssue(
                    "FAIL",
                    "CONFIG_NOT_FOUND",
                    f"{label} does not exist: {path}",
                    remediation="Pass the correct --paths-config/--runtime-config path.",
                )
            )
        elif not os.access(path, os.R_OK):
            issues.append(
                PreflightIssue("FAIL", "CONFIG_NOT_READABLE", f"{label} is not readable: {path}", remediation="Fix permissions or choose another config.")
            )
    env = dict(os.environ)
    env.update({key: str(value) for key, value in base_env.items() if value is not None})

    for config in (paths_config, runtime_config):
        for name, message in required_variables_from_config(config).items():
            if not env.get(name):
                issues.append(
                    PreflightIssue(
                        "FAIL",
                        "CONFIG_REQUIRED_VARIABLE",
                        f"{name} is required by {config}.",
                        details=message or None,
                        remediation="Supply it on the Fett CLI when possible, or export it before running preflight.",
                    )
                )
    if any(issue.severity == "FAIL" and issue.code in {"CONFIG_NOT_FOUND", "CONFIG_NOT_READABLE"} for issue in issues):
        return env, issues

    script = f"""
set -euo pipefail
set -a
source {shell_quote_env(paths_config)}
source {shell_quote_env(runtime_config)}
set +a
python - <<'PY'
import json, os
print(json.dumps(dict(os.environ), sort_keys=True))
PY
"""
    try:
        completed = runner(
            ["bash", "-lc", script],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        issues.append(PreflightIssue("FAIL", "CONFIG_EVALUATION_TIMEOUT", f"Config evaluation timed out after {timeout}s.", details=str(exc)))
        return env, issues
    except OSError as exc:
        issues.append(PreflightIssue("FAIL", "CONFIG_EVALUATION_ERROR", "Could not evaluate shell configs.", details=str(exc)))
        return env, issues
    if completed.returncode != 0:
        issues.append(
            PreflightIssue(
                "FAIL",
                "CONFIG_EVALUATION_FAILED",
                "The configured env files failed in an isolated shell.",
                details=(completed.stderr or completed.stdout).strip(),
                remediation="Fix required variables, syntax errors, or failed source statements in the selected configs.",
            )
        )
        return env, issues
    try:
        resolved = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        issues.append(PreflightIssue("FAIL", "CONFIG_EVALUATION_PARSE_FAILED", "Could not parse resolved config environment.", details=str(exc)))
        return env, issues
    return {str(key): str(value) for key, value in resolved.items()}, issues


def check_path_value(report: PreflightReport, name: str, value: object) -> None:
    text = str(value or "")
    if not text:
        return
    if has_placeholder(text):
        report.fail("PLACEHOLDER_VALUE", f"{name} contains a placeholder value.", details=text, remediation="Replace placeholders with a real path/value.")
    if has_unresolved_expansion(text):
        report.fail("UNRESOLVED_EXPANSION", f"{name} contains an unresolved shell variable.", details=text, remediation="Resolve the value before submission.")
    if has_unmatched_braces(text):
        report.fail("MALFORMED_PATH", f"{name} contains unmatched braces or malformed interpolation.", details=text, remediation="Fix the config interpolation.")
    if text != "/" and "//" in text.replace("://", "SCHEME://"):
        report.warn("SUSPICIOUS_PATH", f"{name} contains repeated slash characters.", details=text)


def check_existing_file(report: PreflightReport, name: str, value: object, *, executable: bool = False, required: bool = True) -> None:
    if not value:
        if required:
            report.fail("MISSING_PATH", f"{name} is not configured.", remediation=f"Set {name} in the runtime config or CLI.")
        return
    check_path_value(report, name, value)
    path = Path(str(value)).expanduser()
    if executable:
        ok = path_is_executable(path)
    else:
        ok = path.is_file()
    if ok:
        report.pass_(f"{name}_OK", f"{name} exists: {path}")
    elif required:
        report.fail("PATH_NOT_FOUND", f"{name} is missing or invalid: {path}", remediation="Install/configure the dependency or select the correct runtime config.")
    else:
        report.warn("OPTIONAL_PATH_NOT_FOUND", f"{name} is missing: {path}")


def check_existing_dir(report: PreflightReport, name: str, value: object, *, required: bool = True, creatable: bool = False) -> None:
    if not value:
        if required:
            report.fail("MISSING_DIRECTORY", f"{name} is not configured.", remediation=f"Set {name} in the runtime config or CLI.")
        return
    check_path_value(report, name, value)
    path = Path(str(value)).expanduser()
    if path.is_dir():
        report.pass_(f"{name}_OK", f"{name} exists: {path}")
    elif creatable:
        parent = path.parent
        if parent.exists() and os.access(parent, os.W_OK):
            report.pass_(f"{name}_CREATABLE", f"{name} can be created: {path}")
        else:
            report.fail("DIRECTORY_NOT_CREATABLE", f"{name} cannot be created: {path}", remediation="Create the parent directory or adjust permissions.")
    elif required:
        report.fail("DIRECTORY_NOT_FOUND", f"{name} does not exist: {path}", remediation="Create/configure the directory before submission.")
    else:
        report.warn("OPTIONAL_DIRECTORY_NOT_FOUND", f"{name} does not exist: {path}")


def check_command(report: PreflightReport, command: str, *, required: bool = True) -> None:
    resolved = shutil.which(command)
    if resolved:
        report.pass_("COMMAND_FOUND", f"{command} available at {resolved}")
    elif required:
        report.fail("COMMAND_NOT_FOUND", f"{command} is not available on PATH.", remediation=f"Install {command} or activate the expected environment.")
    else:
        report.warn("COMMAND_NOT_FOUND", f"{command} is not available on PATH.")


def run_probe(report: PreflightReport, code: str, argv: Sequence[str], *, required: bool = True, timeout: int = 15, runner: Runner = subprocess.run) -> None:
    try:
        completed = runner(list(argv), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except Exception as exc:  # noqa: BLE001 - report preflight exceptions as diagnostics.
        if required:
            report.fail(code, f"Probe failed to start: {' '.join(map(str, argv))}", details=str(exc))
        else:
            report.warn(code, f"Probe failed to start: {' '.join(map(str, argv))}", details=str(exc))
        return
    detail = (completed.stdout or completed.stderr or "").strip().splitlines()
    first = detail[0][:180] if detail else "ok"
    if completed.returncode == 0:
        report.pass_(code, f"Probe passed: {' '.join(map(str, argv[:2]))}", details=first)
    elif required:
        report.fail(code, f"Probe failed: {' '.join(map(str, argv))}", details=first)
    else:
        report.warn(code, f"Probe failed: {' '.join(map(str, argv))}", details=first)


def validate_tarball_input(report: PreflightReport, input_dir: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        report.fail("INPUT_DIR_NOT_FOUND", f"Input directory does not exist: {input_dir}", remediation="Pass an existing --input-dir.")
        return
    if not os.access(input_dir, os.R_OK):
        report.fail("INPUT_DIR_NOT_READABLE", f"Input directory is not readable: {input_dir}", remediation="Fix permissions.")
        return
    report.pass_("INPUT_DIR_READABLE", f"Input directory readable: {input_dir}")
    samples = [path for path in input_dir.iterdir() if path.is_file() and any(path.name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)]
    if not samples:
        report.fail(
            "TARBALL_INPUT_EMPTY",
            f"No supported archive files found in tarball input directory: {input_dir}",
            details=f"Supported suffixes: {', '.join(ARCHIVE_SUFFIXES)}",
            remediation="Point --input-dir at the SAbDab tarball directory.",
        )
        return
    empty = [path.name for path in samples[:10] if path.stat().st_size == 0]
    if empty:
        report.fail("TARBALL_INPUT_INVALID", "Tarball input contains empty archive files.", details=", ".join(empty))
    else:
        report.pass_("TARBALL_INPUT_ARCHIVES", f"Found {len(samples)} supported archive file(s).", details=str(samples[0]))


def validate_output_root(report: PreflightReport, output_root: Path, repo_root: Path, run_id: str) -> None:
    try:
        resolved = output_root.expanduser().resolve()
        source = repo_root.expanduser().resolve()
        if resolved == source or source in resolved.parents:
            report.fail("OUTPUT_ROOT_INSIDE_REPO", f"Output root is inside the source repository: {resolved}", remediation="Use <JANGO_USER_ROOT>/work/<run>.")
            return
    except OSError as exc:
        report.fail("OUTPUT_ROOT_INVALID", f"Output root cannot be resolved: {output_root}", details=str(exc))
        return
    parent = resolved if resolved.exists() else resolved.parent
    while not parent.exists() and parent.parent != parent:
        parent = parent.parent
    if parent.exists() and os.access(parent, os.W_OK):
        report.pass_("OUTPUT_ROOT_WRITABLE", f"Output root can be used: {resolved}")
    else:
        report.fail("OUTPUT_ROOT_NOT_WRITABLE", f"Output root parent is not writable: {resolved}", remediation="Choose a writable /data/$USER location.")
    manifest = resolved / "data" / "manifests" / "fett_run_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            report.fail("OUTPUT_ROOT_MANIFEST_INVALID", f"Existing run manifest is not valid JSON: {manifest}")
        else:
            existing = str(data.get("run_id", ""))
            if existing and existing != run_id:
                report.fail(
                    "OUTPUT_ROOT_RUN_CONFLICT",
                    f"Output root already contains run_id {existing!r}, not {run_id!r}.",
                    remediation="Use a new output root or resume the existing run.",
                )


def validate_decoy_config(report: PreflightReport, path: Path) -> None:
    if has_placeholder(str(path)):
        report.fail("PLACEHOLDER_VALUE", "Decoy config path is a placeholder.", details=str(path), remediation="Pass a real --decoy-config path.")
        return
    if not path.exists() or not path.is_file():
        report.fail("DECOY_CONFIG_NOT_FOUND", f"Decoy config is missing: {path}", remediation="Create/select the antigen redesign decoy config.")
        return
    if not os.access(path, os.R_OK):
        report.fail("DECOY_CONFIG_NOT_READABLE", f"Decoy config is not readable: {path}")
        return
    text = path.read_text(errors="ignore")
    if not text.strip():
        report.fail("DECOY_CONFIG_EMPTY", f"Decoy config is empty: {path}")
    elif has_placeholder(text):
        report.fail("DECOY_CONFIG_PLACEHOLDER", f"Decoy config contains placeholder text: {path}", remediation="Replace TODO/placeholder values.")
    else:
        report.pass_("DECOY_CONFIG_OK", f"Decoy config readable: {path}")


def validate_numeric_options(report: PreflightReport, *, max_structures: int | None = None, case_count: int | None = None, max_decoy_sequences: int) -> None:
    value = max_structures if max_structures is not None else case_count
    if value is None or value <= 0:
        report.fail("INVALID_MAX_STRUCTURES", "--max-structures must be positive.", remediation="Use a positive integer.")
    else:
        report.pass_("MAX_STRUCTURES_OK", f"max_structures={value}")
    if max_decoy_sequences <= 0:
        report.fail("INVALID_MAX_DECOY_SEQUENCES", "--max-decoy-sequences must be positive.", remediation="Use a positive integer.")
    else:
        report.pass_("MAX_DECOY_SEQUENCES_OK", f"max_decoy_sequences={max_decoy_sequences}")


def validate_runtime_dependencies(report: PreflightReport, env: Mapping[str, str], *, backend: str, runner: Runner = subprocess.run) -> None:
    for command in ["fett", "nanobody-pipeline", "decoy-prefold", "decoy-fold", "decoy-analyze", "jango", "sbatch", "squeue", "scontrol"]:
        check_command(report, command, required=command not in {"scontrol"})
    check_existing_dir(report, "PROTEINMPNN_HOME", env.get("PROTEINMPNN_HOME"))
    check_existing_file(report, "PROTEINMPNN_PYTHON", env.get("PROTEINMPNN_PYTHON"), executable=True)
    check_existing_file(report, "PROTEINMPNN_RUNNER", env.get("PROTEINMPNN_RUNNER"))
    if env.get("PROTEINMPNN_PYTHON"):
        run_probe(report, "PROTEINMPNN_PYTHON_PROBE", [env["PROTEINMPNN_PYTHON"], "-c", "import sys; print(sys.executable)"], runner=runner)

    if backend == "esmfold2":
        check_existing_file(report, "ESMFOLD2_PYTHON", env.get("ESMFOLD2_PYTHON"), executable=True)
        check_existing_dir(report, "ESM_ROOT", env.get("ESM_ROOT"))
        check_existing_dir(report, "ESMFOLD2_CACHE", env.get("ESMFOLD2_CACHE"))
        if env.get("ESMC_MODEL_PATH"):
            check_existing_dir(report, "ESMC_MODEL_PATH", env.get("ESMC_MODEL_PATH"))
        if env.get("ESMFOLD2_PYTHON"):
            run_probe(report, "ESMFOLD2_PYTHON_PROBE", [env["ESMFOLD2_PYTHON"], "-c", "import sys; print(sys.version.split()[0])"], runner=runner)
    elif backend == "boltz2":
        check_existing_file(report, "BOLTZ_EXECUTABLE", env.get("BOLTZ_EXECUTABLE") or env.get("BOLTZ"), executable=True)
        check_existing_file(report, "BOLTZ_PYTHON", env.get("BOLTZ_PYTHON"), executable=True, required=False)
        check_existing_dir(report, "BOLTZ_MODEL_ROOT", env.get("BOLTZ_MODEL_ROOT"))
        check_existing_dir(report, "BOLTZ_CACHE", env.get("BOLTZ_CACHE"), creatable=True)
    elif backend == "opendde":
        check_existing_file(report, "OPENDDE_PYTHON", env.get("OPENDDE_PYTHON"), executable=True)
        check_existing_file(report, "OPENDDE_EXECUTABLE", env.get("OPENDDE_EXECUTABLE") or env.get("OPENDDE_BIN"), executable=True)
        check_existing_dir(report, "OPENDDE_ROOT_DIR", env.get("OPENDDE_ROOT_DIR") or env.get("OPENDDE_MODEL_ROOT"))
        check_existing_file(report, "OPENDDE_CHECKPOINT", env.get("OPENDDE_CHECKPOINT"), required=False)
        if env.get("OPENDDE_PYTHON"):
            run_probe(report, "OPENDDE_PYTHON_PROBE", [env["OPENDDE_PYTHON"], "-c", "import sys; print(sys.version.split()[0])"], runner=runner)
    else:
        report.fail("UNSUPPORTED_BACKEND", f"Unsupported backend: {backend}", remediation="Use esmfold2, boltz2, or opendde.")

    if str(env.get("MSA_REQUIRE", "")).lower() in {"1", "true", "yes"}:
        check_existing_dir(report, "MSA_CACHE_ROOT", env.get("MSA_CACHE_ROOT"))
    if str(env.get("MSA_BUILD_IF_MISSING", "")).lower() in {"1", "true", "yes"}:
        check_existing_file(report, "MSA_SCRIPT", env.get("MSA_SCRIPT"))
        check_command(report, "colabfold_search", required=True)
        check_command(report, "mmseqs", required=True)
    elif env.get("MSA_SCRIPT"):
        check_existing_file(report, "MSA_SCRIPT", env.get("MSA_SCRIPT"), required=False)

    check_existing_file(report, "APPTAINER_BIN", env.get("APPTAINER_BIN"), executable=True, required=False)
    check_existing_dir(report, "APPTAINER_CACHEDIR", env.get("APPTAINER_CACHEDIR"), creatable=True, required=False)
    check_existing_file(report, "DOCKQ_PYTHON", env.get("DOCKQ_PYTHON"), executable=True, required=False)
    check_existing_file(report, "ANARCI_PYTHON", env.get("ANARCI_PYTHON"), executable=True, required=False)
    check_existing_file(report, "ANARCI_BIN", env.get("ANARCI_BIN"), executable=True, required=False)


def validate_slurm_resources(report: PreflightReport, env: Mapping[str, str], *, runner: Runner = subprocess.run) -> None:
    gpus = str(env.get("SLURM_GPUS", "0"))
    cpus = str(env.get("SLURM_CPUS_PER_TASK", ""))
    mem = str(env.get("SLURM_MEM", ""))
    time = str(env.get("SLURM_TIME", ""))
    partition = str(env.get("SLURM_PARTITION", ""))
    if not partition:
        report.fail("SLURM_PARTITION_MISSING", "SLURM_PARTITION is not configured.")
    if not re.fullmatch(r"\d+", gpus):
        report.fail("SLURM_GPUS_INVALID", f"SLURM_GPUS must be an integer, got {gpus!r}.")
    if not re.fullmatch(r"\d+", cpus) or int(cpus or "0") <= 0:
        report.fail("SLURM_CPUS_INVALID", f"SLURM_CPUS_PER_TASK must be positive, got {cpus!r}.")
    if not re.fullmatch(r"\d+[KMGTP]?", mem, re.IGNORECASE):
        report.fail("SLURM_MEM_INVALID", f"SLURM_MEM has an invalid format: {mem!r}.")
    if not re.fullmatch(r"(\d+-)?\d{1,2}:\d{2}:\d{2}", time):
        report.fail("SLURM_TIME_INVALID", f"SLURM_TIME has an invalid format: {time!r}.")
    if partition and shutil.which("scontrol"):
        try:
            completed = runner(["scontrol", "show", "partition", partition], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
            if completed.returncode == 0:
                report.pass_("SLURM_PARTITION_OK", f"Slurm partition is visible: {partition}")
            else:
                report.warn("SLURM_PARTITION_UNVERIFIED", f"Could not verify Slurm partition {partition!r}.", details=(completed.stderr or completed.stdout).strip())
        except Exception as exc:  # noqa: BLE001
            report.warn("SLURM_PARTITION_UNVERIFIED", f"Could not verify Slurm partition {partition!r}.", details=str(exc))


def validate_generated_scripts(report: PreflightReport, scripts: Iterable[Path], *, runner: Runner = subprocess.run) -> None:
    for script in scripts:
        report.generated_scripts.append(script)
        text = script.read_text(errors="ignore") if script.exists() else ""
        if not script.exists():
            report.fail("SLURM_SCRIPT_MISSING", f"Generated script is missing: {script}")
            continue
        try:
            completed = runner(["bash", "-n", str(script)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
        except Exception as exc:  # noqa: BLE001
            report.fail("SLURM_SCRIPT_SYNTAX_PROBE_FAILED", f"Could not run bash -n for {script}", details=str(exc))
        else:
            if completed.returncode == 0:
                report.pass_("SLURM_SCRIPT_SYNTAX_OK", f"bash -n passed: {script}", stage=script.stem)
            else:
                report.fail("SLURM_SCRIPT_SYNTAX_ERROR", f"bash -n failed: {script}", details=completed.stderr.strip(), stage=script.stem)
        if "set -a\nsource" in text and "export JANGO_RUN_NAME" not in text.split("set -a\nsource", 1)[0]:
            report.fail(
                "SLURM_ENV_ORDER_INVALID",
                f"Generated script sources configs before exporting run-specific values: {script}",
                remediation="Export JANGO_RUN_NAME/JANGO_RAW_TARBALLS/etc. before source.",
                stage=script.stem,
            )
        for required in ["JANGO_RUN_NAME", "JANGO_RUN_ID", "JANGO_OUTPUT_ROOT", "JANGO_RUN_EXPERIMENT", "FOLDING_BACKEND"]:
            if f"export {required}=" not in text:
                report.fail("SLURM_SCRIPT_MISSING_EXPORT", f"{script} does not export {required}.", stage=script.stem)
        for line in text.splitlines():
            if "${" in line and "SLURM_CPUS_PER_TASK" not in line:
                report.fail("SLURM_SCRIPT_UNRESOLVED_TEMPLATE", f"Generated script contains unresolved template: {line}", stage=script.stem)
            if "}" in line and "{" not in line:
                report.fail("SLURM_SCRIPT_MALFORMED_BRACE", f"Generated script contains malformed brace: {line}", stage=script.stem)
            if "/jango/cache/apptainer" in line and line.count("/jango/cache/apptainer") > 1:
                report.fail("SLURM_SCRIPT_DUPLICATED_PATH", f"Generated script contains duplicated Apptainer cache fragment: {line}", stage=script.stem)


def validate_resolved_config(report: PreflightReport, env: Mapping[str, str], *, cli_backend: str | None = None) -> None:
    report.resolved_config = {name: str(value) for name, value in env.items() if name.startswith("JANGO_") or name in PATH_ENV_NAMES or name.startswith("SLURM_") or name.startswith("FETT_") or name in {"FOLDING_BACKEND", "MSA_REQUIRE", "MSA_BUILD_IF_MISSING", "MSA_PROVIDER"}}
    for name, value in report.resolved_config.items():
        if name in PATH_ENV_NAMES or name.startswith("JANGO_"):
            check_path_value(report, name, value)
            if has_suspicious_path(value):
                report.fail("MALFORMED_PATH", f"{name} is malformed after config evaluation.", details=value, remediation="Fix the config interpolation.")
    if cli_backend and env.get("FOLDING_BACKEND") and env["FOLDING_BACKEND"] != cli_backend:
        report.fail(
            "BACKEND_CONFLICT",
            f"CLI backend {cli_backend!r} conflicts with runtime config FOLDING_BACKEND={env['FOLDING_BACKEND']!r}.",
            remediation="Use the matching runtime config or remove the conflicting --backend.",
        )


def validate_git_state(report: PreflightReport, repo_root: Path, *, runner: Runner = subprocess.run) -> None:
    try:
        completed = runner(["git", "status", "--porcelain"], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
    except Exception as exc:  # noqa: BLE001
        report.warn("GIT_STATUS_UNAVAILABLE", "Could not inspect git status.", details=str(exc))
        return
    if completed.returncode != 0:
        report.warn("GIT_STATUS_UNAVAILABLE", "Could not inspect git status.", details=(completed.stderr or completed.stdout).strip())
    elif completed.stdout.strip():
        report.warn("GIT_DIRTY", "Git working tree has uncommitted changes.", details=completed.stdout.strip().splitlines()[0])
    else:
        report.pass_("GIT_CLEAN", "Git working tree is clean.")

