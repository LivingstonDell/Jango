from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_ESMFOLD2_MODEL = "Biohub/ESMFold2"
DEFAULT_ESMFOLD2_DEVICE = "cuda"
DEFAULT_ROSETTA_IMAGE = "rosettacommons/rosetta:latest"
DEFAULT_DOCKER_IMAGE = DEFAULT_ROSETTA_IMAGE
ROSETTA_RUNTIMES = ("docker", "apptainer", "local")
SOURCE_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH_ATTRS = {
    "manifest": ("out", "checksums"),
    "features-native": ("out",),
    "rosetta-relax": ("work_dir", "status_out"),
    "rosetta-interface": ("work_dir", "out"),
    "rosetta-fullscore-backfill": ("work_dir", "out"),
    "atlas": ("out", "figures_dir"),
    "compare-relaxation": ("out_delta", "out_summary", "figures_dir"),
    "boltz2-cases": ("out",),
    "boltz2-prepare": ("work_dir", "out"),
    "boltz2-analyze": ("tables_dir", "work_dir"),
    "boltz2-report": ("figures_dir",),
    "decoy-cases": ("out",),
    "decoy-prepare": ("work_dir", "out"),
    "decoy-validate": ("work_dir", "tables_dir"),
    "decoy-graft": ("work_dir", "out"),
    "decoy-relax": ("work_dir", "out"),
    "decoy-metrics": ("tables_dir", "work_dir"),
    "decoy-dockq": ("work_dir", "out"),
    "decoy-report": ("figures_dir",),
    "vhh-contact-frequency": ("out_dir",),
}


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def env_bool(name: str, default: bool = False) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int | None = None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


def _resolve_cli_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (Path.cwd() / path).expanduser().resolve()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def guard_generated_outputs_outside_source_repo(args: argparse.Namespace) -> None:
    blocked: list[str] = []
    source_root = SOURCE_REPO_ROOT.expanduser().resolve()
    for attr in OUTPUT_PATH_ATTRS.get(args.command, ()):
        value = getattr(args, attr, None)
        if isinstance(value, Path):
            resolved = _resolve_cli_path(value)
            if _is_within(resolved, source_root):
                option = "--" + attr.replace("_", "-")
                blocked.append(f"{option}={value} -> {resolved}")
    if blocked:
        details = "; ".join(blocked)
        raise ValueError(
            "Refusing to write generated NBIA outputs inside the shared source repository. "
            "Use explicit output paths below <JANGO_USER_ROOT>/<run>. " + details
        )


def add_rosetta_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rosetta-runtime", choices=ROSETTA_RUNTIMES, default=env_value("ROSETTA_RUNTIME"), help="Rosetta runtime: docker, apptainer, or local.")
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"), help="Container image for docker/apptainer runtimes, for example docker://rosettacommons/rosetta:latest.")
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"), help="Directory containing local Rosetta executables when --rosetta-runtime local is used.")
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"), help="Optional Apptainer cache directory.")
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image; also implies --rosetta-runtime docker when runtime is omitted.")


def rosetta_runtime_config_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace, *, required: bool = True):
    from .config import RosettaRuntimeConfig

    runtime = getattr(args, "rosetta_runtime", None)
    docker_image = getattr(args, "docker_image", None)
    if runtime is None and docker_image:
        runtime = "docker"
    if runtime is None:
        if required:
            parser.error("--rosetta-runtime is required for Rosetta execution")
        return None
    image = getattr(args, "rosetta_image", None) or docker_image
    config = RosettaRuntimeConfig(
        kind=runtime,
        image=image,
        bin_dir=getattr(args, "rosetta_bin_dir", None),
        apptainer_cache_dir=getattr(args, "apptainer_cache_dir", None),
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nbia", description="Nanobody interface atlas pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="Build source manifest and checksums")
    p_manifest.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_manifest.add_argument("--out", type=Path, default=Path("data/manifest/manifest.csv"))
    p_manifest.add_argument("--checksums", type=Path, default=Path("data/manifest/checksums.sha256"))

    p_features = sub.add_parser("features-native", help="Compute native interface features")
    p_features.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_features.add_argument("--out", type=Path, default=Path("results/tables/native_interface_features.csv"))
    p_features.add_argument("--contact-cutoff", type=float, default=5.0)

    p_relax = sub.add_parser("rosetta-relax", help="Run constrained interface relax through the selected Rosetta runtime")
    p_relax.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_relax.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_relax.add_argument("--work-dir", type=Path, default=Path("work"))
    add_rosetta_runtime_args(p_relax)
    p_relax.add_argument("--limit", type=int)
    p_relax.add_argument("--jobs", type=int, default=1)
    p_relax.add_argument("--status-out", type=Path, default=Path("results/tables/rosetta_relax_status.csv"))

    p_interface = sub.add_parser("rosetta-interface", help="Run InterfaceAnalyzer through the selected Rosetta runtime")
    p_interface.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_interface.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_interface.add_argument("--work-dir", type=Path, default=Path("work"))
    p_interface.add_argument("--out", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_interface.add_argument("--state", choices=["native", "relaxed"], default="native")
    add_rosetta_runtime_args(p_interface)
    p_interface.add_argument("--limit", type=int)
    p_interface.add_argument("--jobs", type=int, default=1)

    p_fullscore = sub.add_parser("rosetta-fullscore-backfill", help="Backfill missing regular Rosetta score terms")
    p_fullscore.add_argument("--interface-csv", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_fullscore.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_fullscore.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_fullscore.add_argument("--work-dir", type=Path, default=Path("work"))
    p_fullscore.add_argument("--out", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    add_rosetta_runtime_args(p_fullscore)
    p_fullscore.add_argument("--limit", type=int)
    p_fullscore.add_argument("--jobs", type=int, default=1)
    p_fullscore.add_argument("--state", choices=["native", "relaxed"], default="native")

    p_atlas = sub.add_parser("atlas", help="Build joined atlas tables and figures")
    p_atlas.add_argument("--native", type=Path, default=Path("results/tables/native_interface_features.csv"))
    p_atlas.add_argument("--out", type=Path, default=Path("results/tables/interface_features_combined.csv"))
    p_atlas.add_argument("--figures-dir", type=Path, default=Path("results/figures"))

    p_compare = sub.add_parser("compare-relaxation", help="Compare native and relaxed Rosetta interface metrics")
    p_compare.add_argument("--native", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_compare.add_argument("--relaxed", type=Path, default=Path("results/tables/rosetta_interface_relaxed.csv"))
    p_compare.add_argument("--out-delta", type=Path, default=Path("results/tables/rosetta_relaxation_deltas.csv"))
    p_compare.add_argument("--out-summary", type=Path, default=Path("results/tables/rosetta_relaxation_summary.csv"))
    p_compare.add_argument("--figures-dir", type=Path, default=Path("results/figures"))

    p_boltz_cases = sub.add_parser("boltz2-cases", help="Select diverse cases for the Boltz2 benchmark")
    p_boltz_cases.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_boltz_cases.add_argument("--native-features", type=Path, default=Path("results/tables/native_interface_features.csv"))
    p_boltz_cases.add_argument("--rosetta-native", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_boltz_cases.add_argument("--out", type=Path, default=Path("results/tables/boltz2_case_manifest.csv"))
    p_boltz_cases.add_argument("--config", type=Path, default=Path("configs/boltz2.yml"))
    p_boltz_cases.add_argument("--n-cases", type=int, default=12)
    p_boltz_cases.add_argument("--antigen-length-min", type=int, default=60)
    p_boltz_cases.add_argument("--antigen-length-max", type=int, default=250)

    p_boltz_prepare = sub.add_parser("boltz2-prepare", help="Generate Boltz2 YAML inputs and external job manifests")
    p_boltz_prepare.add_argument("--case-manifest", type=Path, default=Path("results/tables/boltz2_case_manifest.csv"))
    p_boltz_prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_boltz_prepare.add_argument("--work-dir", type=Path, default=Path("work/boltz2"))
    p_boltz_prepare.add_argument("--out", type=Path, default=Path("results/tables/boltz2_run_manifest.csv"))
    p_boltz_prepare.add_argument("--limit", type=int)

    p_boltz_analyze = sub.add_parser("boltz2-analyze", help="Analyze returned external Boltz2 prediction outputs")
    p_boltz_analyze.add_argument("--run-manifest", type=Path, default=Path("results/tables/boltz2_run_manifest.csv"))
    p_boltz_analyze.add_argument("--case-manifest", type=Path, default=Path("results/tables/boltz2_case_manifest.csv"))
    p_boltz_analyze.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_boltz_analyze.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_boltz_analyze.add_argument("--work-dir", type=Path, default=Path("work/boltz2"))
    add_rosetta_runtime_args(p_boltz_analyze)
    p_boltz_analyze.add_argument("--skip-rosetta", action="store_true")

    p_boltz_report = sub.add_parser("boltz2-report", help="Build Boltz2 benchmark figures")
    p_boltz_report.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_boltz_report.add_argument("--figures-dir", type=Path, default=Path("results/figures"))

    p_decoy_cases = sub.add_parser("decoy-cases", help="Build antigen redesign decoy case/contact manifest")
    p_decoy_cases.add_argument("--case-manifest", type=Path, default=Path("results/tables/boltz2_case_manifest.csv"))
    p_decoy_cases.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_cases.add_argument("--out", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_cases.add_argument("--config", type=Path, default=Path("configs/production/antigen_redesign_decoys.yml"))
    p_decoy_cases.add_argument("--limit", type=int)

    p_decoy_prepare = sub.add_parser("decoy-prepare", help="Prepare ProteinMPNN inputs and decoy job bundles")
    p_decoy_prepare.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_prepare.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    #added redesign mode argument
    p_decoy_prepare.add_argument("--redesign-mode", choices=["full_antigen", "interface_only", "hotspot_only"], default="full_antigen")
    p_decoy_prepare.add_argument("--out", type=Path, default=Path("results/tables/decoy_mpnn_job_manifest.csv"))
    p_decoy_prepare.add_argument("--limit", type=int)

    p_decoy_validate = sub.add_parser("decoy-validate", help="Parse ProteinMPNN designs and monomer folding validation outputs")
    p_decoy_validate.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_validate.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_validate.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    p_decoy_validate.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_decoy_validate.add_argument("--fold-backend", choices=["boltz2", "esmfold2", "opendde"], default="boltz2")
    p_decoy_validate.add_argument("--validation-phase", choices=["full", "sequence_only"], default="full")
    p_decoy_validate.add_argument("--esmfold2-python", type=Path, default=env_path("ESMFOLD2_PYTHON"))
    p_decoy_validate.add_argument("--esmfold2-root", type=Path, default=env_path("ESM_ROOT"))
    p_decoy_validate.add_argument("--esmfold2-model-id-or-path", default=env_value("ESMFOLD2_MODEL", DEFAULT_ESMFOLD2_MODEL))
    p_decoy_validate.add_argument("--esmfold2-cache-dir", type=Path, default=env_path("ESMFOLD2_CACHE"))
    p_decoy_validate.add_argument("--esmfold2-device", default=env_value("ESMFOLD2_DEVICE", DEFAULT_ESMFOLD2_DEVICE))
    p_decoy_validate.add_argument("--esmfold2-num-sampling-steps", type=int, default=env_int("ESMFOLD2_NUM_SAMPLING_STEPS"))
    p_decoy_validate.add_argument("--esmfold2-num-diffusion-samples", type=int, default=env_int("ESMFOLD2_NUM_DIFFUSION_SAMPLES", 1))
    p_decoy_validate.add_argument("--esmfold2-seed", type=int, default=env_int("ESMFOLD2_SEED"))
    p_decoy_validate.add_argument("--opendde-python", type=Path, default=env_path("OPENDDE_PYTHON"))
    p_decoy_validate.add_argument("--opendde-executable", type=Path, default=env_path("OPENDDE_EXECUTABLE") or env_path("OPENDDE_BIN"))
    p_decoy_validate.add_argument("--opendde-root-dir", type=Path, default=env_path("OPENDDE_ROOT_DIR") or env_path("OPENDDE_MODEL_ROOT"))
    p_decoy_validate.add_argument("--opendde-model-name", default=env_value("OPENDDE_MODEL_NAME"))
    p_decoy_validate.add_argument("--opendde-checkpoint", type=Path, default=env_path("OPENDDE_CHECKPOINT"))
    p_decoy_validate.add_argument("--opendde-seeds", default=env_value("OPENDDE_SEEDS"))
    p_decoy_validate.add_argument("--opendde-cycle", default=env_value("OPENDDE_CYCLE"))
    p_decoy_validate.add_argument("--opendde-step", default=env_value("OPENDDE_STEP"))
    p_decoy_validate.add_argument("--opendde-samples", default=env_value("OPENDDE_SAMPLES"))
    p_decoy_validate.add_argument("--opendde-dtype", default=env_value("OPENDDE_DTYPE"))
    p_decoy_validate.add_argument("--msa-mode", choices=["required", "optional", "disabled", "backend-managed"], default="disabled")
    p_decoy_validate.add_argument("--msa-provider", choices=["precomputed", "abforge_get_or_build", "mmseqs2", "backend-managed", "none"], default=None)
    p_decoy_validate.add_argument("--msa-cache-dir", type=Path, default=None)
    p_decoy_validate.add_argument("--msa-input", type=Path, default=None)
    p_decoy_validate.add_argument("--msa-format", choices=["a3m", "fasta", "csv", "boltz-csv", "stockholm", "unknown"], default=None)
    p_decoy_validate.add_argument("--msa-database", default=None)
    p_decoy_validate.add_argument("--msa-max-sequences", type=int, default=None)
    p_decoy_validate.add_argument("--msa-pairing", choices=["auto", "paired", "unpaired", "none"], default="none")
    p_decoy_validate.add_argument("--msa-reuse-policy", "--msa-design-policy", dest="msa_reuse_policy", default=env_value("MSA_DESIGN_POLICY", env_value("MSA_REUSE_POLICY", "exact_sequence")))
    p_decoy_validate.add_argument("--msa-script", type=Path, default=env_path("MSA_SCRIPT"))
    p_decoy_validate.add_argument("--msa-tool", default=env_value("MSA_TOOL", "esmfold2"))
    p_decoy_validate.add_argument("--msa-build-if-missing", action="store_true", default=env_bool("MSA_BUILD_IF_MISSING"))
    p_decoy_validate.add_argument("--msa-failure-policy", choices=["abort", "skip_design", "skip_case"], default=env_value("MSA_FAILURE_POLICY", "abort"))
    p_decoy_validate.add_argument("--allow-single-sequence-fallback", action="store_true")
    p_decoy_validate.add_argument("--experiment-namespace", default=None)
    p_decoy_validate.add_argument("--max-decoy-sequences", type=int, default=5, help="Maximum ProteinMPNN redesigned sequences sent to folding per structure/mode; native controls are added separately.")
    p_decoy_validate.add_argument("--mode-policy", default=None)

    p_decoy_graft = sub.add_parser("decoy-graft", help="Graft refolded antigen monomers into nanobody poses (A=antigen, N=nanobody)")
    p_decoy_graft.add_argument("--validation", type=Path, default=Path("results/tables/decoy_validation.csv"))
    p_decoy_graft.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_graft.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_graft.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    p_decoy_graft.add_argument("--out", type=Path, default=Path("results/tables/decoy_grafted_complex_manifest.csv"))

    p_decoy_relax = sub.add_parser("decoy-relax", help="Rosetta constrained interface relax of grafted decoy complexes")
    p_decoy_relax.add_argument("--manifest", type=Path, default=Path("results/tables/decoy_grafted_complex_manifest.csv"))
    p_decoy_relax.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    p_decoy_relax.add_argument("--out", type=Path, default=Path("results/tables/decoy_relax_manifest.csv"))
    add_rosetta_runtime_args(p_decoy_relax)
    p_decoy_relax.add_argument("--limit", type=int)
    p_decoy_relax.add_argument("--jobs", type=int, default=1)

    p_decoy_metrics = sub.add_parser("decoy-metrics", help="Score relaxed decoys with the native-crystal metric suite")
    p_decoy_metrics.add_argument("--relax-manifest", type=Path, default=Path("results/tables/decoy_relax_manifest.csv"))
    p_decoy_metrics.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_metrics.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_decoy_metrics.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    add_rosetta_runtime_args(p_decoy_metrics)
    p_decoy_metrics.add_argument("--native-rosetta", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_decoy_metrics.add_argument("--native-protein-interface", type=Path, default=Path("results/tables/native_interface_features.csv"))
    p_decoy_metrics.add_argument("--native-shape-complementarity", type=Path, default=Path("results/tables/native_shape_complementarity.csv"))
    p_decoy_metrics.add_argument("--skip-rosetta", action="store_true")
    p_decoy_metrics.add_argument("--skip-protein-interface", action="store_true")
    p_decoy_metrics.add_argument("--skip-sc", action="store_true")

    p_decoy_dockq = sub.add_parser("decoy-dockq", help="Score relaxed decoys against relaxed native references with official DockQ")
    p_decoy_dockq.add_argument("--decoy-relax-manifest", type=Path, default=Path("results/tables/decoy_relax_manifest.csv"))
    p_decoy_dockq.add_argument("--native-relax-manifest", type=Path, required=True)
    p_decoy_dockq.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_dockq.add_argument("--dockq-bin", type=Path, default=env_path("DOCKQ_BIN"))
    p_decoy_dockq.add_argument("--dockq-provider", choices=["dockq_cli", "dockq_rs_python"], default=env_value("DOCKQ_PROVIDER", "dockq_cli"))
    p_decoy_dockq.add_argument("--dockq-python", type=Path, default=env_path("DOCKQ_PYTHON"))
    p_decoy_dockq.add_argument("--work-dir", type=Path, default=Path("work/dockq"))
    p_decoy_dockq.add_argument("--out", type=Path, default=Path("results/tables/decoy_dockq.csv"))
    p_decoy_dockq.add_argument("--jobs", type=int, default=1)
    p_decoy_dockq.add_argument("--limit", type=int)
    p_decoy_dockq.add_argument("--allow-failures", action="store_true")
    p_decoy_dockq.add_argument("--skip-existing", action="store_true")
    p_decoy_dockq.add_argument("--experiment-namespace", default=None)

    p_decoy_report = sub.add_parser("decoy-report", help="Build antigen redesign decoy figures")
    p_decoy_report.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_decoy_report.add_argument("--figures-dir", type=Path, default=Path("results/figures"))

    p_vhh_contacts = sub.add_parser("vhh-contact-frequency", help="Analyze ANARCI-numbered VHH antigen-contact frequencies")
    p_vhh_contacts.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_vhh_contacts.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_vhh_contacts.add_argument("--out-dir", type=Path, default=Path("analyses/vhh_contact_frequency_4p5A"))
    p_vhh_contacts.add_argument("--contact-cutoff", type=float, default=4.5)
    p_vhh_contacts.add_argument("--limit", type=int)

    args = parser.parse_args(argv)
    try:
        guard_generated_outputs_outside_source_repo(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "manifest":
        from .manifest import build_manifest

        df = build_manifest(args.raw_dir, args.out, args.checksums)
        print(f"wrote {len(df)} manifest rows to {args.out}")
    elif args.command == "features-native":
        from .features import compute_native_features

        df = compute_native_features(args.raw_dir, args.out, args.contact_cutoff)
        print(f"wrote {len(df)} feature rows to {args.out}")
    elif args.command == "rosetta-relax":
        from .rosetta import run_rosetta_relax

        config = rosetta_runtime_config_from_args(parser, args)
        df = run_rosetta_relax(args.manifest, args.raw_dir, args.work_dir, args.status_out, docker_image=None, limit=args.limit, jobs=args.jobs, runtime_config=config)
        print(f"wrote {len(df)} relax status rows to {args.status_out}")
    elif args.command == "rosetta-interface":
        from .rosetta import run_interface_analyzer

        config = rosetta_runtime_config_from_args(parser, args)
        df = run_interface_analyzer(args.manifest, args.raw_dir, args.work_dir, args.out, state=args.state, docker_image=None, limit=args.limit, jobs=args.jobs, runtime_config=config)
        print(f"wrote {len(df)} InterfaceAnalyzer rows to {args.out}")
    elif args.command == "rosetta-fullscore-backfill":
        from .rosetta import run_fullscore_backfill

        config = rosetta_runtime_config_from_args(parser, args)
        df = run_fullscore_backfill(
            args.interface_csv,
            args.manifest,
            args.raw_dir,
            args.work_dir,
            args.out,
            docker_image=None,
            limit=args.limit,
            jobs=args.jobs,
            state=args.state,
            runtime_config=config,
        )
        print(f"wrote {len(df)} rows with backfilled regular score terms to {args.out}")
    elif args.command == "atlas":
        from .plots import build_atlas_tables_and_figures

        df = build_atlas_tables_and_figures(args.native, args.out, args.figures_dir)
        print(f"wrote atlas for {len(df)} structures")
    elif args.command == "compare-relaxation":
        from .compare import compare_rosetta_native_relaxed

        paired, summary = compare_rosetta_native_relaxed(args.native, args.relaxed, args.out_delta, args.out_summary, args.figures_dir)
        print(f"wrote {len(paired)} paired relaxation rows and {len(summary)} metric summaries")
    elif args.command == "boltz2-cases":
        from .boltz2 import select_boltz2_cases

        df = select_boltz2_cases(
            args.manifest,
            args.native_features,
            args.rosetta_native,
            args.out,
            args.config,
            args.n_cases,
            args.antigen_length_min,
            args.antigen_length_max,
        )
        print(f"wrote {len(df)} Boltz2 benchmark cases to {args.out}")
    elif args.command == "boltz2-prepare":
        from .boltz2 import prepare_boltz2_inputs

        df = prepare_boltz2_inputs(args.case_manifest, args.raw_dir, args.work_dir, args.out, args.limit)
        print(f"wrote {len(df)} Boltz2 run rows to {args.out}")
    elif args.command == "boltz2-analyze":
        from .boltz2 import analyze_boltz2_outputs

        config = rosetta_runtime_config_from_args(parser, args, required=not args.skip_rosetta)
        prediction, quality, recovery, rosetta, summary = analyze_boltz2_outputs(
            args.run_manifest,
            args.case_manifest,
            args.raw_dir,
            args.tables_dir,
            args.work_dir,
            not args.skip_rosetta,
            docker_image=None,
            runtime_config=config,
        )
        print(
            f"wrote Boltz2 analysis tables: {len(prediction)} predictions, "
            f"{len(quality)} quality rows, {len(recovery)} recovery rows, {len(rosetta)} Rosetta rows, {len(summary)} summary rows"
        )
    elif args.command == "boltz2-report":
        from .boltz2 import build_boltz2_report

        build_boltz2_report(args.tables_dir, args.figures_dir)
        print(f"wrote Boltz2 report figures to {args.figures_dir}")
    elif args.command == "decoy-cases":
        from .decoys import build_decoy_cases

        df = build_decoy_cases(args.case_manifest, args.raw_dir, args.out, args.config, args.limit)
        print(f"wrote {len(df)} decoy redesign cases to {args.out}")
    elif args.command == "decoy-prepare":
        from .decoys import prepare_decoy_inputs

        df = prepare_decoy_inputs(args.case_manifest, args.raw_dir, args.work_dir, args.out, args.limit, args.redesign_mode)
        print(f"wrote {len(df)} ProteinMPNN job rows to {args.out}")
    elif args.command == "decoy-validate":
        from .decoys import validate_decoy_designs

        candidates, fold_jobs, validation = validate_decoy_designs(
            args.case_manifest,
            args.raw_dir,
            args.work_dir,
            args.tables_dir,
            fold_backend=args.fold_backend,
            esmfold2_python=args.esmfold2_python,
            esmfold2_root=args.esmfold2_root,
            esmfold2_model_id_or_path=args.esmfold2_model_id_or_path,
            esmfold2_cache_dir=args.esmfold2_cache_dir,
            esmfold2_device=args.esmfold2_device,
            esmfold2_num_sampling_steps=args.esmfold2_num_sampling_steps,
            esmfold2_num_diffusion_samples=args.esmfold2_num_diffusion_samples,
            esmfold2_seed=args.esmfold2_seed,
            opendde_python=args.opendde_python,
            opendde_executable=args.opendde_executable,
            opendde_root_dir=args.opendde_root_dir,
            opendde_model_name=args.opendde_model_name,
            opendde_checkpoint=args.opendde_checkpoint,
            opendde_seeds=args.opendde_seeds,
            opendde_cycle=args.opendde_cycle,
            opendde_step=args.opendde_step,
            opendde_samples=args.opendde_samples,
            opendde_dtype=args.opendde_dtype,
            msa_mode=args.msa_mode,
            msa_provider=args.msa_provider,
            msa_cache_dir=args.msa_cache_dir,
            msa_input=args.msa_input,
            msa_format=args.msa_format,
            msa_database=args.msa_database,
            msa_max_sequences=args.msa_max_sequences,
            msa_pairing=args.msa_pairing,
            msa_reuse_policy=args.msa_reuse_policy,
            msa_script=args.msa_script,
            msa_tool=args.msa_tool,
            msa_build_if_missing=args.msa_build_if_missing,
            msa_failure_policy=args.msa_failure_policy,
            allow_single_sequence_fallback=args.allow_single_sequence_fallback,
            experiment_namespace=args.experiment_namespace,
            max_decoy_sequences_per_structure=args.max_decoy_sequences,
            mode_policy=args.mode_policy,
            validation_phase=args.validation_phase,
        )
        print(
            f"wrote {len(candidates)} candidates, {len(fold_jobs)} {args.fold_backend} monomer jobs, "
            f"and {len(validation)} validation rows"
        )
    elif args.command == "decoy-graft":
        from .decoys import graft_decoy_complexes

        df = graft_decoy_complexes(args.validation, args.case_manifest, args.raw_dir, args.work_dir, args.out)
        ok = int((df["graft_status"] == "ok").sum()) if "graft_status" in df.columns and not df.empty else 0
        print(f"wrote {len(df)} grafted decoy rows ({ok} ok) to {args.out}")
    elif args.command == "decoy-relax":
        from .rosetta import run_decoy_relax

        config = rosetta_runtime_config_from_args(parser, args)
        df = run_decoy_relax(args.manifest, args.work_dir, args.out, docker_image=None, limit=args.limit, jobs=args.jobs, runtime_config=config)
        print(f"wrote {len(df)} decoy relax rows to {args.out}")
    elif args.command == "decoy-metrics":
        from .decoys import compute_decoy_metrics

        config = rosetta_runtime_config_from_args(parser, args, required=not args.skip_rosetta)
        out = compute_decoy_metrics(
            args.relax_manifest,
            args.case_manifest,
            args.tables_dir,
            args.work_dir,
            not args.skip_rosetta,
            not args.skip_protein_interface,
            not args.skip_sc,
            docker_image=None,
            runtime_config=config,
            native_rosetta_csv=args.native_rosetta,
            native_protein_interface_csv=args.native_protein_interface,
            native_shape_complementarity_csv=args.native_shape_complementarity,
        )
        print(f"wrote decoy metrics: {len(out['comparison'])} comparison rows to {args.tables_dir}/decoy_native_comparison.csv")
    elif args.command == "decoy-dockq":
        from .quality.dockq import DockQError, score_decoy_manifest

        try:
            df = score_decoy_manifest(
                decoy_relax_manifest=args.decoy_relax_manifest,
                native_relax_manifest=args.native_relax_manifest,
                case_manifest=args.case_manifest,
                dockq_bin=args.dockq_bin,
                dockq_provider=args.dockq_provider,
                dockq_python=args.dockq_python,
                work_dir=args.work_dir,
                out_csv=args.out,
                jobs=args.jobs,
                limit=args.limit,
                allow_failures=args.allow_failures,
                skip_existing=args.skip_existing,
                experiment_namespace=args.experiment_namespace,
            )
        except DockQError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        failures = int((df["dockq_status"] != "ok").sum()) if "dockq_status" in df.columns and not df.empty else 0
        print(f"wrote {len(df)} official DockQ rows to {args.out} ({failures} failures)")
    elif args.command == "decoy-report":
        from .decoys import build_decoy_report

        build_decoy_report(args.tables_dir, args.figures_dir)
        print(f"wrote decoy report figures to {args.figures_dir}")
    elif args.command == "vhh-contact-frequency":
        from .vhh_contacts import AnarciUnavailableError, run_vhh_contact_frequency_analysis

        try:
            numbering, contacts, frequency, summary = run_vhh_contact_frequency_analysis(args.manifest, args.raw_dir, args.out_dir, args.contact_cutoff, args.limit)
        except AnarciUnavailableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            f"wrote VHH contact-frequency analysis to {args.out_dir}: "
            f"{len(numbering)} numbered residues, {len(contacts)} contact residues, {len(frequency)} ANARCI positions, {len(summary)} structures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







