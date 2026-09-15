# Jango Onboarding Lesson Plan

This lesson plan is for a first-time Jango user learning the product in short sessions. It assumes <SLURM_NODE> and the shared installation at `<JANGO_REPO>`.

## 1. What Jango Is For

- Scientific question: generate antigen decoys that preserve structure while perturbing predicted nanobody binding.
- Difference between sequence-only folding and full structure-aware decoy validation.
- Why native complex context is required for grafting, DockQ, and comparable metrics.

## 2. What You Need Before Starting

- Native complex PDBs or SAbDab-style tarballs.
- Antigen and nanobody chain metadata.
- Optional custom sequence/mutation table.
- Optional MSA cache, depending on backend.
- A manifest linking sequences back to native structures and chain roles.

## 3. Environment Setup On gnode

Users should not manually export every runtime path. The shared Conda environment installs a production activation hook, so startup is:

```bash
ssh <SLURM_NODE>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango
cd <JANGO_REPO>
jango doctor --quick
```

Activation sets shared defaults such as `JANGO_SOURCE`, user workspace roots, shared tarballs, shared MSA cache, ESMFold2 default backend, ProteinMPNN, Rosetta, DockQ-RS, ANARCI, and Slurm settings.

For modular commands, select a run explicitly:

```bash
jango-use-run my_run_id
```

To switch backend defaults interactively:

```bash
jango-use-backend esmfold2
jango-use-backend opendde
jango-use-backend boltz2
```

## 4. Workspace Layout

- Source: `<JANGO_REPO>`
- User work: `<JANGO_USER_ROOT>/work/<run-id>`
- User outputs: `<JANGO_USER_ROOT>/outputs`
- User cache/tmp: `<JANGO_USER_ROOT>/cache`, `<JANGO_USER_ROOT>/tmp`
- Curated shared data: `<JANGO_REPO>/data`

## 5. Fett Command

Teach Fett as the normal production entry point:

```bash
fett preflight --input /path/to/raw/data --run-id my_run --max-structures 50
fett submit    --input /path/to/raw/data --run-id my_run --max-structures 50
fett status my_run
fett logs my_run
fett resume my_run
fett cancel my_run
fett export-final my_run
```

`fett` resolves run-specific paths, configs, and Slurm staging. A user does not need `jango-use-run` before Fett.

## 6. Full Pipeline

Explain each stage in order:

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

Focus on what each stage consumes, writes, and validates.

## 7. Modular `jango` Commands

Teach modular commands after Fett:

```bash
jango decoy-prefold --help
jango decoy-fold --help
jango fold-jobs --help
jango fold-qc --help
jango decoy-analyze --help
jango native-refold-compare --help
jango reference-benchmark-figures --help
jango high-high-figures --help
jango lineage-audit --help
```

Emphasize that modular commands need a selected run/context and valid manifests.

## 8. Inputs And Redesign Modes

- Hotspot only: mutates selected high-impact residues.
- Interface only: mutates antigen residues near the nanobody interface.
- Full antigen: allows redesign across the antigen.
- Custom sequence inputs: replace ProteinMPNN only if they are linked to native structures by manifest.
- Sequence-only inputs: can fold monomers, but cannot complete full Jango decoy validation.

## 9. Folding Backends

- ESMFold2: default interactive backend.
- ESMFold2 sampling mode: ESMFold2-based sampling workflow when configured.
- OpenDDE: diffusion-style folding backend with multiple generated samples.
- Boltz2: configured runtime surface, used when selected.

Teach backend switching with `jango-use-backend` and backend-specific validation with `jango doctor --quick --backend <backend>`.

## 10. Reading Outputs

Prioritize:

- final relaxed decoy PDBs
- native relaxed references
- native refold controls
- top-1/fold provenance manifests
- ranked ML dataset tables
- high-high figures
- DockQ component figures

## 11. Interpreting Metrics

- TM-score: antigen fold preservation.
- CA RMSD: antigen coordinate deviation.
- DockQ: interface/complex preservation, not fold confidence.
- DockQ components: Fnat, iRMSD, LRMSD.
- Rosetta delta dG: computational scoring feature, not reliable experimental affinity.
- Binding perturbation: Jango-derived feature used by the original quadrant logic.

## 12. Debugging And Recovery

- Run `jango doctor --quick` after activation.
- Run `jango-use-run <run-id>` before modular commands.
- Check Slurm with `squeue -u $USER`.
- Inspect logs under run-local `logs/`.
- Use lineage/path audits to catch stale `/home`, `<TOOLS_ROOT>`, `<CPU_NODE>`, or old validation paths.
- Resume when possible; rerun only failed shards/stages when manifests make that safe.

## 13. Best Practices

- Keep generated work out of source unless it is curated shared data.
- Use descriptive run IDs.
- Preserve final PDBs, ranked tables, provenance, configs, and logs.
- Treat bulky intermediates as regenerable after validation.
- Move validated user data to shared data deliberately, not during active jobs.

## 14. Known Limitations And Next Steps

- Rosetta dG does not match literature/experimental affinity reliably enough to be interpreted as binding truth.
- SKEMPI-style exact mutation validation is needed to calibrate binding perturbation against experimental delta delta G.
- Custom sequence import should become a first-class command that builds canonical sequence manifests.
- Full-antigen redesign needs the same validated treatment as hotspot and interface modes before broad scientific claims.
