"""Canonical run directory layout for Jango workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def repository_root() -> Path:
    """Return the installed source repository root."""

    return Path(__file__).resolve().parents[2]


def resolve_output_root(output_root: Path, *, repo_root: Path | None = None) -> Path:
    """Resolve and validate a user-selected run output root."""

    root = output_root.expanduser().resolve()
    source_root = (repo_root or repository_root()).expanduser().resolve()
    if root == source_root or source_root in root.parents:
        raise ValueError(f"--output-root must not be inside the canonical repository: {source_root}")
    return root


def recommended_run_root(run_name: str, *, test: bool = False, user: str | None = None) -> Path:
    """Return the conventional per-user Jango run root for a named run."""

    account = user or os.environ.get("USER") or os.environ.get("USERNAME")
    if not account:
        raise ValueError("could not determine user name for recommended run root")
    return Path("/data") / account / "jango" / "work" / run_name


@dataclass(frozen=True)
class RawSourcePaths:
    """Raw input locations for one source/dataset label."""

    root: Path

    @property
    def pdb(self) -> Path:
        return self.root / "pdb"

    @property
    def cif(self) -> Path:
        return self.root / "cif"

    @property
    def tarballs(self) -> Path:
        return self.root / "tarballs"


@dataclass(frozen=True)
class DecoyLayout:
    """Canonical artifact paths for one decoy experiment branch inside a run."""

    run: "RunPaths"
    experiment_key: str

    @property
    def manifest_dir(self) -> Path:
        return self.run.manifests_decoy / _safe_component(self.experiment_key)

    @property
    def prefold_manifest(self) -> Path:
        return self.manifest_dir / "prefold_manifest.json"

    @property
    def source_case_manifest(self) -> Path:
        return self.manifest_dir / "source_case_manifest.csv"

    @property
    def base_decoy_case_manifest(self) -> Path:
        return self.manifest_dir / "decoy_redesign_case_manifest.base.csv"

    @property
    def mode_assignments(self) -> Path:
        return self.manifest_dir / "auto_mode_assignments.csv"

    @property
    def fold_manifest(self) -> Path:
        return self.manifest_dir / "fold_manifest.json"

    @property
    def combined_fold_submission_manifest(self) -> Path:
        return self.run.work_decoy / _safe_component(self.experiment_key) / "slurm" / "fold_submission_manifest.csv"

    @property
    def all_modes_tables_dir(self) -> Path:
        return self.run.results_decoy_tables / _safe_component(self.experiment_key) / "all_modes"

    @property
    def all_modes_figures_dir(self) -> Path:
        return self.run.results_decoy_figures / _safe_component(self.experiment_key) / "all_modes"

    @property
    def analysis_dir(self) -> Path:
        return self.run.results_decoy / "analysis" / _safe_component(self.experiment_key) / "all_modes"

    @property
    def analysis_tables_dir(self) -> Path:
        return self.analysis_dir / "tables"

    def mode_manifest_dir(self, mode: str) -> Path:
        return self.manifest_dir / _safe_component(mode)

    def mode_case_manifest(self, mode: str) -> Path:
        return self.mode_manifest_dir(mode) / "decoy_redesign_case_manifest.csv"

    def mode_work_dir(self, mode: str) -> Path:
        return self.run.work_decoy / _safe_component(self.experiment_key) / _safe_component(mode)

    def mode_tables_dir(self, mode: str) -> Path:
        return self.run.results_decoy_tables / _safe_component(self.experiment_key) / _safe_component(mode)

    def mode_figures_dir(self, mode: str) -> Path:
        return self.run.results_decoy_figures / _safe_component(self.experiment_key) / _safe_component(mode)

    def mode_validation(self, mode: str) -> Path:
        return self.mode_tables_dir(mode) / "decoy_validation.csv"

    def mode_grafted_manifest(self, mode: str) -> Path:
        return self.mode_tables_dir(mode) / "decoy_grafted_complex_manifest.csv"

    def mode_relax_manifest(self, mode: str) -> Path:
        return self.mode_tables_dir(mode) / "decoy_relax_manifest.csv"

    def mode_interface_table(self, mode: str) -> Path:
        return self.mode_tables_dir(mode) / "decoy_rosetta_interface.csv"

    def mode_dockq_table(self, mode: str) -> Path:
        return self.mode_tables_dir(mode) / "decoy_dockq.csv"


@dataclass(frozen=True)
class RunPaths:
    """Directory model for all generated files in one Jango run."""

    root: Path

    @classmethod
    def from_output_root(cls, output_root: Path, *, repo_root: Path | None = None) -> "RunPaths":
        return cls(resolve_output_root(output_root, repo_root=repo_root))

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def work_native(self) -> Path:
        return self.work / "native"

    @property
    def work_decoy(self) -> Path:
        return self.work / "decoy"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def manifests_native(self) -> Path:
        return self.manifests / "native"

    @property
    def manifests_decoy(self) -> Path:
        return self.manifests / "decoy"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def results_native(self) -> Path:
        return self.results / "native"

    @property
    def results_native_tables(self) -> Path:
        return self.results_native / "tables"

    @property
    def results_native_figures(self) -> Path:
        return self.results_native / "figures"

    @property
    def results_decoy(self) -> Path:
        return self.results / "decoy"

    @property
    def results_decoy_tables(self) -> Path:
        return self.results_decoy / "tables"

    @property
    def results_decoy_figures(self) -> Path:
        return self.results_decoy / "figures"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def raw_source(self, source: str) -> RawSourcePaths:
        return RawSourcePaths(self.data_raw / _safe_component(source))

    def decoy_layout(self, experiment: str) -> DecoyLayout:
        return DecoyLayout(self, _safe_component(experiment))

    def decoy_manifest_dir(self, experiment: str, mode: str | None = None) -> Path:
        layout = self.decoy_layout(experiment)
        return layout.mode_manifest_dir(mode) if mode else layout.manifest_dir

    def decoy_work_dir(self, experiment: str, mode: str) -> Path:
        return self.decoy_layout(experiment).mode_work_dir(mode)

    def decoy_tables_dir(self, experiment: str, mode: str) -> Path:
        return self.decoy_layout(experiment).mode_tables_dir(mode)

    def decoy_figures_dir(self, experiment: str, mode: str) -> Path:
        return self.decoy_layout(experiment).mode_figures_dir(mode)


def _safe_component(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("path component must be non-empty")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"unsafe path component: {value!r}")
    return text