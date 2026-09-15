from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

    p_relax = sub.add_parser("rosetta-relax", help="Run constrained interface relax through Docker")
    p_relax.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_relax.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_relax.add_argument("--work-dir", type=Path, default=Path("work"))
    p_relax.add_argument("--docker-image", default="rosettacommons/rosetta:latest")
    p_relax.add_argument("--limit", type=int)
    p_relax.add_argument("--jobs", type=int, default=1)
    p_relax.add_argument("--status-out", type=Path, default=Path("results/tables/rosetta_relax_status.csv"))

    p_interface = sub.add_parser("rosetta-interface", help="Run InterfaceAnalyzer through Docker")
    p_interface.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_interface.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_interface.add_argument("--work-dir", type=Path, default=Path("work"))
    p_interface.add_argument("--out", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_interface.add_argument("--state", choices=["native", "relaxed"], default="native")
    p_interface.add_argument("--docker-image", default="rosettacommons/rosetta:latest")
    p_interface.add_argument("--limit", type=int)
    p_interface.add_argument("--jobs", type=int, default=1)

    p_fullscore = sub.add_parser("rosetta-fullscore-backfill", help="Backfill missing regular Rosetta score terms")
    p_fullscore.add_argument("--interface-csv", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_fullscore.add_argument("--manifest", type=Path, default=Path("data/manifest/manifest.csv"))
    p_fullscore.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_fullscore.add_argument("--work-dir", type=Path, default=Path("work"))
    p_fullscore.add_argument("--out", type=Path, default=Path("results/tables/rosetta_interface_native.csv"))
    p_fullscore.add_argument("--docker-image", default="rosettacommons/rosetta:latest")
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
    p_boltz_analyze.add_argument("--docker-image", default="rosettacommons/rosetta:latest")
    p_boltz_analyze.add_argument("--skip-rosetta", action="store_true")

    p_boltz_report = sub.add_parser("boltz2-report", help="Build Boltz2 benchmark figures")
    p_boltz_report.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_boltz_report.add_argument("--figures-dir", type=Path, default=Path("results/figures"))

    p_decoy_cases = sub.add_parser("decoy-cases", help="Build antigen redesign decoy case/contact manifest")
    p_decoy_cases.add_argument("--case-manifest", type=Path, default=Path("results/tables/boltz2_case_manifest.csv"))
    p_decoy_cases.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_cases.add_argument("--out", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_cases.add_argument("--config", type=Path, default=Path("configs/antigen_redesign_decoys.yml"))
    p_decoy_cases.add_argument("--limit", type=int)

    p_decoy_prepare = sub.add_parser("decoy-prepare", help="Prepare ProteinMPNN inputs and decoy job bundles")
    p_decoy_prepare.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_prepare.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    p_decoy_prepare.add_argument("--out", type=Path, default=Path("results/tables/decoy_mpnn_job_manifest.csv"))
    p_decoy_prepare.add_argument("--limit", type=int)

    p_decoy_validate = sub.add_parser("decoy-validate", help="Parse ProteinMPNN designs and Boltz2 monomer validation outputs")
    p_decoy_validate.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_validate.add_argument("--raw-dir", type=Path, default=Path("data/raw/pdb"))
    p_decoy_validate.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    p_decoy_validate.add_argument("--tables-dir", type=Path, default=Path("results/tables"))

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
    p_decoy_relax.add_argument("--docker-image", default="rosettacommons/rosetta:latest")
    p_decoy_relax.add_argument("--limit", type=int)
    p_decoy_relax.add_argument("--jobs", type=int, default=1)

    p_decoy_metrics = sub.add_parser("decoy-metrics", help="Score relaxed decoys with the native-crystal metric suite")
    p_decoy_metrics.add_argument("--relax-manifest", type=Path, default=Path("results/tables/decoy_relax_manifest.csv"))
    p_decoy_metrics.add_argument("--case-manifest", type=Path, default=Path("results/tables/decoy_redesign_case_manifest.csv"))
    p_decoy_metrics.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p_decoy_metrics.add_argument("--work-dir", type=Path, default=Path("work/decoys"))
    p_decoy_metrics.add_argument("--docker-image", default="rosettacommons/rosetta:latest")
    p_decoy_metrics.add_argument("--skip-rosetta", action="store_true")
    p_decoy_metrics.add_argument("--skip-protein-interface", action="store_true")
    p_decoy_metrics.add_argument("--skip-sc", action="store_true")

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

        df = run_rosetta_relax(args.manifest, args.raw_dir, args.work_dir, args.status_out, args.docker_image, args.limit, args.jobs)
        print(f"wrote {len(df)} relax status rows to {args.status_out}")
    elif args.command == "rosetta-interface":
        from .rosetta import run_interface_analyzer

        df = run_interface_analyzer(args.manifest, args.raw_dir, args.work_dir, args.out, args.state, args.docker_image, args.limit, args.jobs)
        print(f"wrote {len(df)} InterfaceAnalyzer rows to {args.out}")
    elif args.command == "rosetta-fullscore-backfill":
        from .rosetta import run_fullscore_backfill

        df = run_fullscore_backfill(
            args.interface_csv,
            args.manifest,
            args.raw_dir,
            args.work_dir,
            args.out,
            args.docker_image,
            args.limit,
            args.jobs,
            args.state,
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

        prediction, quality, recovery, rosetta, summary = analyze_boltz2_outputs(
            args.run_manifest,
            args.case_manifest,
            args.raw_dir,
            args.tables_dir,
            args.work_dir,
            not args.skip_rosetta,
            args.docker_image,
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

        df = prepare_decoy_inputs(args.case_manifest, args.raw_dir, args.work_dir, args.out, args.limit)
        print(f"wrote {len(df)} ProteinMPNN job rows to {args.out}")
    elif args.command == "decoy-validate":
        from .decoys import validate_decoy_designs

        candidates, boltz, validation = validate_decoy_designs(args.case_manifest, args.raw_dir, args.work_dir, args.tables_dir)
        print(f"wrote {len(candidates)} candidates, {len(boltz)} Boltz2 monomer jobs, and {len(validation)} validation rows")
    elif args.command == "decoy-graft":
        from .decoys import graft_decoy_complexes

        df = graft_decoy_complexes(args.validation, args.case_manifest, args.raw_dir, args.work_dir, args.out)
        ok = int((df["graft_status"] == "ok").sum()) if "graft_status" in df.columns and not df.empty else 0
        print(f"wrote {len(df)} grafted decoy rows ({ok} ok) to {args.out}")
    elif args.command == "decoy-relax":
        from .rosetta import run_decoy_relax

        df = run_decoy_relax(args.manifest, args.work_dir, args.out, args.docker_image, args.limit, args.jobs)
        print(f"wrote {len(df)} decoy relax rows to {args.out}")
    elif args.command == "decoy-metrics":
        from .decoys import compute_decoy_metrics

        out = compute_decoy_metrics(
            args.relax_manifest,
            args.case_manifest,
            args.tables_dir,
            args.work_dir,
            not args.skip_rosetta,
            not args.skip_protein_interface,
            not args.skip_sc,
            args.docker_image,
        )
        print(f"wrote decoy metrics: {len(out['comparison'])} comparison rows to {args.tables_dir}/decoy_native_comparison.csv")
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
