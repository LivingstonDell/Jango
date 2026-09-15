# Pipeline

The canonical user-facing stages are split by purpose:

- `nanobody-pipeline`: native input preparation, native manifests/features, Rosetta relax, and relaxed native InterfaceAnalyzer scoring.
- `decoy-prefold`: source case selection, NBIA decoy contact manifest generation, and redesign-mode manifest split.
- `decoy-fold`: internal staged decoy preparation helpers for ProteinMPNN job preparation, CPU ProteinMPNN execution, sequence validation, MSA resolution/native-MSA reuse, and fold bundle generation.
- `fold-jobs`: Slurm fold submission/status/resume helpers for generated fold bundles.
- `fold-qc`: backend-neutral post-fold QC over existing fold manifests and outputs.
- `decoy-analyze`: grafting, required decoy Rosetta scoring, DockQ-RS against relaxed native references, native-decoy comparison, figures, landscape tables, quadrant ranking, and mutation localization.
- `rosetta-qc`: post-Rosetta QC for existing native and decoy Rosetta outputs.
- `fett`: production-facing orchestration that resolves installation defaults, preflights the full run, submits the resume-safe Slurm chain, tracks dependencies, and exports ML-ready tables.


Fett resource allocation is automatic and installation-owned. Normal users do not provide CPU, memory, GPU, or wall-time settings. The current Slurm chain is:

```text
native_analysis
-> decoy_prepare
-> proteinmpnn
-> sequence_validation
-> msa_resolution
-> decoy_folding
-> fold_qc
-> decoy_metrics_analysis
```

Only `decoy_folding` reserves one GPU. ProteinMPNN, MSA resolution, sequence validation, fold QC, native analysis, and decoy metrics/analysis run CPU-only. No Fett stage requests more than 24G RAM.

Normal Fett usage:

```bash
fett preflight --input /path/to/raw/data --run-id validation_50 --max-structures 50
fett submit --input /path/to/raw/data --run-id validation_50 --max-structures 50
fett status validation_50
fett logs validation_50
fett resume validation_50
fett cancel validation_50
```

The canonical user workspace is:

```text
<JANGO_USER_ROOT>/
+-- runs/<run-id>/
+-- cache/{msa,apptainer,metadata}/
+-- registry/
+-- tmp/
```

Each new run stores root-level reproducibility files:

```text
<run-root>/
+-- run.json
+-- resolved_config.json
+-- provenance.json
+-- inputs/input_manifest.json
+-- inputs/decoy_config.yml
+-- data/manifests/fett_run_manifest.json
+-- work/
+-- logs/
+-- results/
+-- exports/
```

Scientific algorithms remain in `nbia`; Jango wrappers pass explicit paths into NBIA and derive generated artifacts from the resolved run root. Existing internal `data/manifests`, `work/decoy`, and `results/decoy` paths are preserved for compatibility.

`fett preflight` validates input detection, run ID safety, workspace permissions, canonical configs, dependency paths, Slurm resources, output-root conflicts, generated scripts, and run snapshots without submitting jobs. `fett submit` and `fett resume` run the same blocking preflight unless `--skip-preflight` is used deliberately.

Advanced long-form commands remain supported for administrators through the `Advanced overrides` help group.
