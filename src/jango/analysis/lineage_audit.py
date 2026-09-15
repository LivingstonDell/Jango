"""Read-only lineage audit for generated Jango artifacts.

The audit walks table/manifest files, extracts path-like references, and reports
missing files, stale roots, paths outside approved roots, and redesign-mode path
mismatches. It never modifies the data it scans.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from jango.paths import repository_root


TABLE_SUFFIXES = {".csv", ".tsv"}
JSON_SUFFIXES = {".json", ".jsonl"}
MODE_NAMES = ("hotspot_only", "interface_only", "full_antigen")
DEFAULT_STALE_ROOTS = (
    "<HOME>",
    "<TOOLS_ROOT>",
)
PATH_NAME_HINTS = (
    "path",
    "pdb",
    "file",
    "dir",
    "manifest",
    "input",
    "output",
    "msa",
    "fasta",
    "csv",
    "tsv",
    "json",
)
PATH_SUFFIXES = (
    ".a3m",
    ".cif",
    ".csv",
    ".fa",
    ".fasta",
    ".json",
    ".jsonl",
    ".pdb",
    ".tar",
    ".tgz",
    ".tsv",
    ".txt",
)


REFERENCE_COLUMNS = [
    "source_file",
    "source_kind",
    "row_index",
    "json_path",
    "column",
    "value",
    "resolved_path",
    "exists",
    "source_mode",
    "value_mode",
    "issues",
]
ISSUE_COLUMNS = [
    "issue_type",
    "source_file",
    "source_kind",
    "row_index",
    "json_path",
    "column",
    "value",
    "resolved_path",
    "source_mode",
    "value_mode",
]
SUMMARY_COLUMNS = [
    "source_file",
    "source_kind",
    "rows",
    "path_references",
    "missing_path",
    "stale_root",
    "outside_allowed_root",
    "mode_mismatch",
    "unreadable_artifact",
]


@dataclass(frozen=True)
class AuditConfig:
    roots: tuple[Path, ...]
    allowed_roots: tuple[Path, ...] = ()
    stale_roots: tuple[str, ...] = DEFAULT_STALE_ROOTS


@dataclass(frozen=True)
class PathReference:
    source_file: Path
    source_kind: str
    row_index: int | None
    json_path: str
    column: str
    value: str
    resolved_path: Path
    exists: bool
    source_mode: str
    value_mode: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactSummary:
    source_file: Path
    source_kind: str
    rows: int
    path_references: int = 0
    missing_path: int = 0
    stale_root: int = 0
    outside_allowed_root: int = 0
    mode_mismatch: int = 0
    unreadable_artifact: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lineage-audit",
        description="Audit generated Jango manifests/tables for broken or stale path lineage.",
    )
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Root to scan. Repeatable. Defaults to the repository data/outputs directory.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for lineage audit TSV reports.")
    parser.add_argument(
        "--allowed-root",
        action="append",
        type=Path,
        default=[],
        help="Approved root for referenced files. Repeatable. If provided, references outside these roots are reported.",
    )
    parser.add_argument(
        "--stale-root",
        action="append",
        default=list(DEFAULT_STALE_ROOTS),
        help="Path prefix that should be reported as stale. Repeatable.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Return non-zero when any lineage issue is found.",
    )
    return parser


def default_roots() -> tuple[Path, ...]:
    return (repository_root() / "data" / "outputs",)


def audit_roots(config: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    references: list[PathReference] = []
    summaries: list[ArtifactSummary] = []
    for artifact in iter_artifacts(config.roots):
        try:
            artifact_refs, summary = audit_artifact(artifact, config)
        except Exception as exc:  # pragma: no cover - defensive report path
            artifact_refs = []
            summary = ArtifactSummary(
                source_file=artifact,
                source_kind=artifact.suffix.lower().lstrip("."),
                rows=0,
                unreadable_artifact=1,
            )
            references.append(
                PathReference(
                    source_file=artifact,
                    source_kind=summary.source_kind,
                    row_index=None,
                    json_path="",
                    column="",
                    value=f"unreadable: {exc}",
                    resolved_path=artifact,
                    exists=artifact.exists(),
                    source_mode=infer_mode(artifact),
                    value_mode="",
                    issues=("unreadable_artifact",),
                )
            )
        references.extend(artifact_refs)
        summaries.append(summary)

    references_df = references_to_frame(references)
    issues_df = issues_to_frame(references)
    summary_df = summaries_to_frame(summaries)
    return references_df, issues_df, summary_df


def iter_artifacts(roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [p for p in root.rglob("*") if p.is_file()]
        else:
            continue
        for path in candidates:
            if path.suffix.lower() not in TABLE_SUFFIXES | JSON_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def audit_artifact(path: Path, config: AuditConfig) -> tuple[list[PathReference], ArtifactSummary]:
    suffix = path.suffix.lower()
    if suffix in TABLE_SUFFIXES:
        refs, rows = audit_table(path, config)
        source_kind = suffix.lstrip(".")
    elif suffix in JSON_SUFFIXES:
        refs, rows = audit_json(path, config)
        source_kind = suffix.lstrip(".")
    else:
        refs, rows, source_kind = [], 0, suffix.lstrip(".")
    return refs, summarize_artifact(path, source_kind, rows, refs)


def audit_table(path: Path, config: AuditConfig) -> tuple[list[PathReference], int]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return [], 0

    refs: list[PathReference] = []
    source_mode = infer_mode(path)
    for column in df.columns:
        series = df[column]
        if not is_candidate_path_column(column, series):
            continue
        for row_index, value in series.items():
            text = clean_text(value)
            if not looks_like_path(text):
                continue
            refs.append(build_reference(path, "table", int(row_index), "", column, text, source_mode, config))
    return refs, len(df)


def audit_json(path: Path, config: AuditConfig) -> tuple[list[PathReference], int]:
    refs: list[PathReference] = []
    source_mode = infer_mode(path)
    if path.suffix.lower() == ".jsonl":
        row_count = 0
        for row_index, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            row_count += 1
            data = json.loads(line)
            refs.extend(walk_json(path, data, f"$[{row_index}]", source_mode, config))
        return refs, row_count

    data = json.loads(path.read_text())
    refs.extend(walk_json(path, data, "$", source_mode, config))
    return refs, 1


def walk_json(path: Path, data: Any, json_path: str, source_mode: str, config: AuditConfig) -> list[PathReference]:
    refs: list[PathReference] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{json_path}.{key}"
            if isinstance(value, str):
                text = clean_text(value)
                if (is_path_name(str(key)) or looks_like_path(text)) and looks_like_path(text):
                    refs.append(build_reference(path, "json", None, child_path, str(key), text, source_mode, config))
            elif isinstance(value, (dict, list)):
                refs.extend(walk_json(path, value, child_path, source_mode, config))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            refs.extend(walk_json(path, value, f"{json_path}[{index}]", source_mode, config))
    return refs


def build_reference(
    source_file: Path,
    source_kind: str,
    row_index: int | None,
    json_path: str,
    column: str,
    value: str,
    source_mode: str,
    config: AuditConfig,
) -> PathReference:
    resolved = resolve_reference(value, source_file)
    exists = resolved.exists()
    value_mode = infer_mode_from_text(value)
    issues = tuple(find_issues(value, resolved, exists, source_mode, value_mode, config))
    return PathReference(source_file, source_kind, row_index, json_path, column, value, resolved, exists, source_mode, value_mode, issues)


def find_issues(
    value: str,
    resolved: Path,
    exists: bool,
    source_mode: str,
    value_mode: str,
    config: AuditConfig,
) -> list[str]:
    issues: list[str] = []
    if not exists:
        issues.append("missing_path")
    if any(value.startswith(root) or str(resolved).startswith(root) for root in config.stale_roots):
        issues.append("stale_root")
    if config.allowed_roots and not any(is_under(resolved, root) for root in config.allowed_roots):
        issues.append("outside_allowed_root")
    if source_mode and value_mode and source_mode != value_mode:
        issues.append("mode_mismatch")
    return issues


def summarize_artifact(path: Path, source_kind: str, rows: int, refs: Sequence[PathReference]) -> ArtifactSummary:
    return ArtifactSummary(
        source_file=path,
        source_kind=source_kind,
        rows=rows,
        path_references=len(refs),
        missing_path=sum("missing_path" in ref.issues for ref in refs),
        stale_root=sum("stale_root" in ref.issues for ref in refs),
        outside_allowed_root=sum("outside_allowed_root" in ref.issues for ref in refs),
        mode_mismatch=sum("mode_mismatch" in ref.issues for ref in refs),
    )


def references_to_frame(references: Sequence[PathReference]) -> pd.DataFrame:
    rows = [
        {
            "source_file": str(ref.source_file),
            "source_kind": ref.source_kind,
            "row_index": "" if ref.row_index is None else ref.row_index,
            "json_path": ref.json_path,
            "column": ref.column,
            "value": ref.value,
            "resolved_path": str(ref.resolved_path),
            "exists": ref.exists,
            "source_mode": ref.source_mode,
            "value_mode": ref.value_mode,
            "issues": ";".join(ref.issues),
        }
        for ref in references
    ]
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


def issues_to_frame(references: Sequence[PathReference]) -> pd.DataFrame:
    rows = []
    for ref in references:
        for issue in ref.issues:
            rows.append(
                {
                    "issue_type": issue,
                    "source_file": str(ref.source_file),
                    "source_kind": ref.source_kind,
                    "row_index": "" if ref.row_index is None else ref.row_index,
                    "json_path": ref.json_path,
                    "column": ref.column,
                    "value": ref.value,
                    "resolved_path": str(ref.resolved_path),
                    "source_mode": ref.source_mode,
                    "value_mode": ref.value_mode,
                }
            )
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def summaries_to_frame(summaries: Sequence[ArtifactSummary]) -> pd.DataFrame:
    rows = [
        {
            "source_file": str(summary.source_file),
            "source_kind": summary.source_kind,
            "rows": summary.rows,
            "path_references": summary.path_references,
            "missing_path": summary.missing_path,
            "stale_root": summary.stale_root,
            "outside_allowed_root": summary.outside_allowed_root,
            "mode_mismatch": summary.mode_mismatch,
            "unreadable_artifact": summary.unreadable_artifact,
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_reports(references: pd.DataFrame, issues: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "references": out_dir / "lineage_references.tsv",
        "issues": out_dir / "lineage_issues.tsv",
        "summary": out_dir / "lineage_summary.tsv",
    }
    references.to_csv(paths["references"], sep="\t", index=False)
    issues.to_csv(paths["issues"], sep="\t", index=False)
    summary.to_csv(paths["summary"], sep="\t", index=False)
    return paths


def is_candidate_path_column(column: str, series: pd.Series) -> bool:
    if is_path_name(column):
        return True
    sample = [clean_text(value) for value in series.head(50)]
    return any(looks_like_path(value) for value in sample)


def is_path_name(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in PATH_NAME_HINTS)


def looks_like_path(value: str) -> bool:
    if not value or value.lower() == "nan" or "://" in value:
        return False
    if "%" in value or any(char.isspace() for char in value):
        return False
    lowered = value.lower()
    if lowered.startswith(("/", "./", "../", "~")):
        return True
    if any(lowered.endswith(suffix) for suffix in PATH_SUFFIXES):
        return True
    return "/" in value


def resolve_reference(value: str, source_file: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_file.parent / path
    return path.resolve()


def clean_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text


def infer_mode(path: Path) -> str:
    return infer_mode_from_parts(path.parts)


def infer_mode_from_text(value: str) -> str:
    return infer_mode_from_parts(tuple(part for part in value.replace("\\", "/").split("/") if part))


def infer_mode_from_parts(parts: Sequence[str]) -> str:
    for part in parts:
        if part in MODE_NAMES:
            return part
    return ""


def is_under(path: Path, root: Path) -> bool:
    root = root.expanduser().resolve()
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def print_report(references: pd.DataFrame, issues: pd.DataFrame, summary: pd.DataFrame, paths: dict[str, Path]) -> None:
    issue_counts = issues["issue_type"].value_counts().to_dict() if not issues.empty else {}
    print(f"audited_artifacts: {len(summary)}")
    print(f"path_references: {len(references)}")
    print(f"issues: {len(issues)}")
    for issue_type in sorted(issue_counts):
        print(f"  {issue_type}: {issue_counts[issue_type]}")
    for label, path in paths.items():
        print(f"wrote_{label}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = tuple(path.expanduser().resolve() for path in (args.roots or list(default_roots())))
    allowed_roots = tuple(path.expanduser().resolve() for path in args.allowed_root)
    config = AuditConfig(roots=roots, allowed_roots=allowed_roots, stale_roots=tuple(args.stale_root))
    references, issues, summary = audit_roots(config)
    paths = write_reports(references, issues, summary, args.out_dir.expanduser().resolve())
    print_report(references, issues, summary, paths)
    return 1 if args.fail_on_issues and not issues.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
