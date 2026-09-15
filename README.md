## Scientific Rationale

Jango is designed to ask a specific question: can we create antigen decoys that preserve the overall antigen/nanobody complex geometry while perturbing predicted binding? In practice, this means generating antigen sequence variants, folding the redesigned antigen, grafting the folded antigen back into the original nanobody complex, relaxing/scoring comparable complexes, and ranking decoys by structural preservation and binding perturbation.

The pipeline uses three complementary ideas:

* **Controlled redesign:** ProteinMPNN proposes sequence variants for user-selected antigen regions: hotspot residues, interface residues, or full antigen redesign.
* **Backend-specific folding:** ESMFold2 or OpenDDE predicts the structure of redesigned antigen sequences. Native refolds are used as backend controls to estimate whether the folding backend preserves antigen structure before interpreting decoys.
* **Comparable complex evaluation:** Folded antigens are grafted back onto the native nanobody context, relaxed with Rosetta, and compared against relaxed references with DockQ/TM-style structural metrics and Rosetta-derived interface scores.

The most useful current visual summaries are Q1/core figures: DockQ versus delta dG, refold TM-score histograms, Q1 refold-quality histograms, reference-bias plots, mutation-profile maps, and backend/design-mode comparisons. DockQ captures complex/interface preservation; TM-score captures antigen fold preservation; delta dG is a Rosetta-derived predicted interface-energy change and must be interpreted cautiously.

## Project Status

Current validated scope: hotspot decoy generation/evaluation with ESM and OpenDDE backends.

| Area | Status | Notes |
| --- | --- | --- |
| SAbDab2 input curation | Done | Curated shared inputs live on <SLURM_NODE> shared storage. |
| ProteinMPNN sequence generation | Done | Hotspot, interface-only, and full-antigen sequence outputs exist; hotspot is the final validated analysis focus. |
| Hotspot ESM folding | Done | Fold/graft/relax/metrics completed for the hotspot package. |
| Hotspot OpenDDE folding | Done | Fold/graft/relax/metrics completed for the hotspot package. |
| Native refold controls | Done | Used to assess backend fold accuracy before interpreting decoys. |
| Reference-bias validation package | Done | Crystal, relaxed, and refold reference comparisons are summarized under `data/outputs/evaluation/validation` and review add-ons under `docs/analysis_addons`. |
| Hotspot core report package | Done | Curated report assets are packaged for GitHub review under `docs/analysis_addons`. |
| Mutation-profile analysis | Partly done | Concise OpenDDE hotspot mutation-profile analysis is report-ready; broader mode/backend mutation interpretation remains exploratory. |
| Interface-only analysis | Generated, not final-report validated | Interface outputs exist, but this README/report centers on hotspot. |
| Full-antigen analysis | Not validated | Full-antigen MPNN outputs exist, but final fold/graft/relax/report analysis was not completed. |
| SKEMPI experimental validation | Not implemented | Needed to test whether Jango/Rosetta features correlate with experimental delta delta G. |
| External user FASTA/CIF onboarding | Partial design only | Jango can operate from manifests; a polished importer for arbitrary user CIF/PDB/FASTA packages is future work. |
| Analysis add-on packages | Done | Reference-bias and concise mutation-profile analyses are separated under `docs/analysis_addons` for review and future canonization. |

## Quick Start

On <SLURM_NODE>, users only need to activate the shared Conda environment. Activation automatically loads the shared Jango defaults through the production activation hook:

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango
cd <JANGO_REPO>

```

After activation, these are already set for ordinary users:

```text
JANGO_SOURCE=<JANGO_REPO>
JANGO_WORK_ROOT=<JANGO_USER_ROOT>/work
JANGO_OUTPUTS_ROOT=<JANGO_USER_ROOT>/outputs
JANGO_DATA_ROOT=<JANGO_USER_ROOT>/data
JANGO_RAW_TARBALLS=<JANGO_REPO>/data/tarballs
FOLDING_BACKEND=esmfold2
MSA_CACHE_ROOT=<JANGO_REPO>/data/msa

```

Run a quick health check:

```bash
jango doctor --quick
fett --help

```

`jango doctor` may warn that `JANGO_RUN_ROOT` is not set if you have not selected a run yet. That is expected for a fresh shell. For modular `jango ...` commands, select a run with:

```bash
jango-use-run my_run_id
jango doctor --quick

```

To switch the interactive backend defaults:

```bash
jango-use-backend opendde   # or esmfold2 / boltz2

```

A standard Fett run starts with preflight, then submit/status/logs. Fett resolves the run-specific paths itself, so you do not need to call `jango-use-run` first:

```bash
fett preflight --input /path/to/raw/data --run-id validation_50 --max-structures 50
fett submit    --input /path/to/raw/data --run-id validation_50 --max-structures 50

fett status validation_50
fett logs validation_50
fett resume validation_50
fett cancel validation_50
fett export-final validation_50

```

`--max-structures` limits the number of input structures at the beginning of the workflow. The default interactive backend is ESMFold2 unless changed with `jango-use-backend` or a backend-specific command/config.

## GitHub Handoff Notes

The GitHub repository should contain code, tests, documentation, and small report assets. Large generated datasets should stay on shared storage or be published as separate artifacts.

Included or suitable for Git:

* `src/`, `scripts/`, `configs/`, `tests/`, and `docs/`.
* `docs/jango_hotspot_final_report.md`.
* `docs/analysis_addons/`, which contains small report-ready add-on packages for exploratory analyses not yet exposed as first-class Jango commands.

Not suitable for normal Git:

* Full fold outputs, relaxed PDBs, raw tarballs, MSA caches, Slurm logs, and run-local `work/` or `tmp/` folders.
* Large curated data under `<JANGO_REPO>/data`; this remains the <SLURM_NODE> source-of-truth data store unless separately packaged.

## Repository And Data Layout

The Git source repository should stay mostly source-only:

```text
<JANGO_REPO>/
+-- src/
|   +-- nbia/          # scientific engine and preserved NBIA CLI
|   +-- jango/         # orchestration, paths, CLI, analysis commands
+-- scripts/           # operational wrappers and compatibility scripts
+-- configs/           # administrator-maintained runtime configs
+-- docs/
+-- tests/
+-- data/              # curated shared input/output data only
+-- README.md

```

The canonical user workspace is:

```text
<JANGO_USER_ROOT>/
+-- cache/             # reusable user caches
+-- work/              # run-local intermediates and logs
+-- outputs/           # user-visible final outputs
+-- tmp/               # disposable diagnostics

```

Shared curated data currently follows:

```text
<JANGO_REPO>/data/
+-- folds/             # folded antigen outputs by backend/mode
+-- mpnn/              # ProteinMPNN sequence outputs by redesign mode
+-- msa/               # canonical active MSA cache used by MSA_CACHE_ROOT
+-- msa_cache/         # legacy compatibility mirror of the MSA cache
+-- outputs/
    +-- evaluation/    # canonical evaluation tables/figures
+-- relaxed/           # relaxed complexes/decoys where curated
+-- tarballs/          # shared raw SAbDab2 tarballs

```

`data/msa` is the runtime MSA cache Jango points to by default. Keep `data/msa_cache` in place for compatibility with older manifests, scripts, and users until a separate migration confirms it can be safely replaced by a symlink or removed. Do not add new top-level data categories casually. Put meaning in the run ID, backend name, redesign mode, and table metadata.

## Canonical Evaluation Packages

Canonical report outputs live under:

```text
<JANGO_REPO>/data/outputs/evaluation/
+-- decoys/
    +-- figures/      # report-facing decoy landscape/comparison figures
    +-- tables/       # ML-ready and ranked decoy tables
    +-- manifest.csv  # links artifacts to source tables and commands
+-- validation/
    +-- figures/      # mutation-profile, reference-bias, and fold-control figures
    +-- tables/       # validation summaries and supporting statistics
    +-- manifest.csv  # links validation artifacts to source tables and commands

```

The evaluation directory is intentionally compact: `decoys/` and
`validation/` are the only report-facing packages. Historical broad export
folders were removed after the core artifacts were promoted, so use the package
manifests to trace figures back to their canonical supporting tables. Do not
create new top-level `data/outputs/evaluation_*` folders for routine analysis.
Q1 is the canonical term for the high-DockQ / positive-delta-dG quadrant.

## Reproducibility Map

Primary <SLURM_NODE> source repository:

```text
<JANGO_REPO>

```

Canonical shared data:

```text
<JANGO_REPO>/data
+-- folds/
+-- mpnn/
+-- msa/
+-- msa_cache/
+-- outputs/
    +-- evaluation/
+-- relaxed/
+-- tarballs/

```

Report-ready hotspot packages on <SLURM_NODE>:

```text
<JANGO_REPO>/data/outputs/evaluation/decoys
<JANGO_REPO>/data/outputs/evaluation/validation

```

GitHub-renderable analysis add-on packages:

```text
docs/analysis_addons

```

Local figure mirror, when synchronized:

```text
<LOCAL_JANGO_COPY>\data\outputs\evaluation\decoys
<LOCAL_JANGO_COPY>\data\outputs\evaluation\validation

```

## Analysis Add-Ons

Some report-ready analyses were performed around canonical Jango outputs but
are not yet first-class reusable Jango commands. These are separated under:

```text
docs/analysis_addons/
+-- hotspot_reference_bias/
+-- hotspot_mutation_profile/

```

This keeps the analyses visible for review without implying that the current
pipeline can regenerate them through a stable CLI endpoint. The intended next
step is to turn each add-on into a dedicated `jango ...` command.

## Pipeline Overview

The full production chain is:

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

Only folding should reserve a GPU. ProteinMPNN, MSA resolution, sequence validation, fold QC, native analysis, Rosetta-side metrics, DockQ, and figure generation are CPU-only in the normal Slurm chain.

Key stages:

* **Native analysis:** creates native manifests, native relaxed structures, and native/reference metrics.
* **Decoy preparation:** identifies redesignable antigen residues and creates redesign-mode manifests.
* **ProteinMPNN:** generates candidate mutant antigen sequences for hotspot, interface-only, or full-antigen redesign.
* **MSA resolution:** links each fold input to the expected MSA cache where required by the backend.
* **Folding:** folds mutant antigen sequences with ESMFold2 or OpenDDE.
* **Fold QC/top-1 selection:** ranks fold outputs per native/design group and selects the canonical best decoy set.
* **Grafting:** replaces the native antigen coordinates in the original complex with the folded redesigned antigen while retaining the nanobody context.
* **Rosetta relax/scoring:** relaxes comparable complexes and computes interface metrics.
* **DockQ/native comparison:** compares decoy complexes to the appropriate relaxed reference.
* **Metrics/figures:** writes compact tables, ML-ready ranked datasets, and canonical Q1/core figures.

## Useful CLI Commands

General health and lineage:

```bash
jango doctor
jango lineage-audit --help

```

Native/refold comparison:

```bash
jango native-refold-compare --help

```

Fold and downstream decoy workflow commands:

```bash
jango decoy-prefold --help
jango decoy-fold --help
jango fold-jobs --help
jango fold-qc --help
jango decoy-analyze --help
jango rosetta-qc --help

```

Canonical figure/table commands:

```bash
jango core-figures --help
jango reference-benchmark-figures --help
jango high-high-figures --help

```

`reference-benchmark-figures` is the canonical command for report-ready DockQ/delta-dG decoy-vs-reference visualization. It writes one organized package with per-reference landscapes, backend comparisons, redesign-mode comparisons, ML-ready ranked tables, summary statistics, and plot provenance. Q1 is the current name for the preserved-structure / perturbed-binding quadrant; `high-high-figures` remains available only as a compatibility command for older exports.

The older scripts remain compatibility wrappers:

```bash
scripts/analysis/export_high_high_figures.py

```

Prefer the `jango ...` commands for new work.

## Outputs To Keep

For handoff, modeling, or ML-ready datasets, prioritize:

* Final relaxed decoy PDBs.
* Native relaxed reference PDBs.
* Native refold relaxed/control structures where backend fold accuracy is part of the analysis.
* Fold provenance and top-1 selection manifests.
* `data/outputs/evaluation/decoys/tables/hotspot_bias_robust_ml_dataset.csv` for the curated hotspot Q1 package, or the mode/backend-specific ranked ML table.
* Core summary CSVs and JSON manifests from the canonical figure commands.
* Slurm logs and resolved configs only when needed for reproducibility/debugging.

Pre-relax grafted PDBs and bulky intermediate fold outputs are usually regenerable if source sequences, fold manifests, selected top-1 records, and references are preserved.

## Interpreting Core Metrics

* **TM-score:** global antigen fold similarity. Jango reports this as a percent for refold-quality plots, with 90% used as the current high-quality cutoff.
* **CA RMSD:** C-alpha coordinate deviation for the antigen comparison. It is useful but more length/geometry sensitive than TM-score.
* **DockQ:** composite interface/complex similarity metric. It combines interface contact preservation and RMSD-like terms; it is not a model-confidence score.
* **DockQ components:** Fnat, iRMSD, and LRMSD help unpack which part of DockQ is driving a result.
* **Rosetta delta dG:** predicted change in separated interface energy. It is useful as a relative computational feature inside a controlled pipeline, but it is not a validated experimental affinity measurement.
* **Binding perturbation:** Jango-derived binding-change feature used by the original quadrant logic. High-high figure sets instead focus directly on DockQ and delta dG.

## Current Validation Pattern

For each backend and redesign mode, the most interpretable package is:

1. Native crystal relaxed versus native refold relaxed: asks whether the backend preserves the native antigen fold.
2. Decoy folded/relaxed versus native refold relaxed: asks whether backend-generated decoys preserve structure while perturbing predicted binding in a backend-consistent coordinate space.
3. Optional decoy folded/relaxed versus crystal relaxed: useful as an external reference, but less controlled if backend refolding introduces systematic structural shifts.

Avoid comparing relaxed structures to unrelaxed structures. Avoid comparing ESMFold2-folded decoys directly to OpenDDE-folded decoys without a matched reference/control design.

## Testing And Validation

Before changing pipeline behavior, run focused tests and audits:

```bash
python -m py_compile src/jango/analysis/high_high_figures.py src/jango/analysis/dockq_component_figures.py src/jango/cli.py src/jango/doctor.py
pytest -q tests/jango/test_canonical_figure_commands.py tests/jango/test_core_figures.py tests/jango/test_native_refold_compare.py tests/jango/test_esmfold2_doctor.py

```

For production-like changes, also run from a fresh activated shell:

```bash
jango doctor --quick
jango-use-run smoke_test
jango doctor --quick
jango lineage-audit --help
fett preflight --input /path/to/raw/data --run-id smoke_test --max-structures 5

```

When debugging path issues, audit generated manifests and tables for stale paths such as `<HOME>`, `<TOOLS_ROOT>`, `<CPU_NODE>`, or old validation run IDs before submitting long jobs.

## Problems, Caveats, And Next Steps

* **Rosetta dG is not experimental affinity.** Current Rosetta interface delta dG scoring does not match literature/experimental affinity well enough to be treated as reliable binding affinity. It should be framed as a computational feature or ranking signal, not as ground-truth delta delta G.
* **SKEMPI validation remains important.** A future validation path should replace ProteinMPNN sequence generation with exact SKEMPI mutation-table sequences for matched complexes, then correlate Jango binding perturbation/Rosetta features with experimental delta delta G.
* **Path lineage must remain strict.** Stale absolute paths have caused repeated downstream failures. Every manifest consumed by a later stage should be auditable, canonical, and free of old home/<CPU_NODE> paths.
* **Top-1 selection must stay backend-transparent.** ESM and OpenDDE should both produce comparable top-1 decoy sets per native/design group, with fold provenance retained.
* **Intermediate retention policy should stay lean.** Keep final structures, ranked tables, provenance, configs, and logs. Delete or archive bulky regenerable intermediates after validation.
* **Full-antigen redesign has not been prioritized.** Current validated analyses focus on hotspot and interface-only modes. Full-antigen mode needs the same fold/graft/relax/metrics validation before scientific interpretation.
* **DockQ component plots are diagnostic, not a new ranking rule.** They explain DockQ behavior but should not silently redefine success criteria without a documented analysis decision.
* **GitHub repo should stay lightweight.** Keep source, docs, tests, and small report assets in Git; keep large curated data on shared storage or package it separately.
* **External user onboarding needs a manifest importer.** A future user should be able to provide CIF/PDB structures plus metadata, or custom FASTA sequences for an already prepared complex, without manually assembling internal manifests.
* **Mutation-profile claims need broader validation.** Current concise chemistry interpretation is strongest for OpenDDE hotspot Q1+TM90 decoys and should not be overgeneralized yet.

## References

Additional implementation details are in:

* `docs/ARCHITECTURE.md`
* `docs/PIPELINE.md`
* `docs/CONFIGURATION.md`
* `docs/DEPLOYMENT.md`
* `src/nbia/folding/README_backend_integration.md`

```

```
