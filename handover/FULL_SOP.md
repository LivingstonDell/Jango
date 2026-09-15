# Jango Full SOP

This SOP explains the full SAbDab2 antigen-redesign project protocol, including why each major design choice was made, what each step consumes and produces, and how to validate or debug the step. It is written for a new project member who needs to reproduce the validated hotspot/interface analyses without relying on private scratch directories.

## Scope

This SOP covers the validated Jango workflow for:

- SAbDab2 nanobody-antigen complexes.
- Hotspot and interface antigen redesign modes.
- ESM and OpenDDE folding backends.
- Top-1 decoy selection per native complex.
- Grafting, Rosetta relax, DockQ, Rosetta interface analysis, mutation profiling, reference-bias analysis, and compact figure/table publication.

This SOP does not cover the unfinished full-antigen analysis or the future SKEMPI validation workflow.

## Ground Rules

- Work from `<JANGO_REPO>` unless explicitly using a disposable user run root.
- Treat `<JANGO_REPO>/data` as canonical shared data.
- Treat `<JANGO_USER_ROOT>` as scratch/raw provenance unless a specific output has been promoted.
- Never move shared MSA data during active jobs.
- Do not edit generated manifests by hand unless followed by a lineage audit.
- Do not interpret Rosetta energy as experimental affinity.
- Public labels use `ESM` and `OpenDDE`.

## Stage 0: Environment and Path Setup

Purpose:
Set one reproducible runtime before touching data or jobs.

Inputs:

- Shared repo: `<JANGO_REPO>`
- Production configs under `configs/production`
- Shared tools under `<TOOLS_ROOT>`
- Shared conda environments under `<CONDA_ROOT>/envs`

Commands:

```bash
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango
jango doctor
```

Expected outputs:

- `jango doctor` reports `FAIL - none`.
- ProteinMPNN, ESM, OpenDDE, Rosetta, DockQ-RS, ANARCI, MSA helper, and Slurm are visible.

Design justification:

- A shared path configuration prevents hidden user-home paths from entering manifests.
- `jango doctor` is the fastest way to catch missing tool paths before a multi-hour Slurm job fails.

Debugging:

- If a tool path is unset, inspect `configs/production/paths.env` and backend-specific env files.
- If Slurm resources are wrong, inspect `configs/production/runtime.<SLURM_NODE>.common.env`.
- If MSA checks warn but `MSA_BUILD_IF_MISSING=false`, this is acceptable for jobs that use existing MSA cache.

## Stage 1: Canonical Data Layout

Purpose:
Keep raw data, work products, and report outputs separated.

Canonical roots:

- `data/mpnn`: ProteinMPNN sequence outputs.
- `data/folds`: folded antigen outputs by backend and redesign mode.
- `data/relaxed`: final relaxed complexes and native-refold relaxed structures.
- `data/outputs/evaluation`: compact report-facing figures and tables.
- `handover`: orientation, reproduction commands, checksums, maps, figures, compact tables, and bundled source tables.

Design justification:

- Heavy PDB directories stay in canonical data roots.
- The evaluation directory stays small and report-facing.
- The handover package carries enough source tables to regenerate figures without calling user scratch directories.

Validation:

```bash
find data/outputs/evaluation -type f -name '*.png' | wc -l
bash handover/commands/validate_handover.sh
```

## Stage 2: Reference Dataset Definition

Purpose:
Define the native structures against which redesigns are interpreted.

Inputs:

- SAbDab2 native nanobody-antigen complexes.
- Metadata containing structure ID, nanobody chain, antigen chain, and source PDB/CIF path.

Reference states:

- Crystal native: experimental coordinate file.
- Relaxed native: crystal native after Rosetta relax.
- Native refold relaxed: native antigen refolded by backend, grafted to the native nanobody context, then relaxed.

Outputs:

- Native manifests.
- Relaxed-native structures.
- Native-refold relaxed structures.
- Native refold quality tables.

Design justification:

- Crystal native is the experimental anchor.
- Relaxed native isolates Rosetta relaxation effects.
- Native refold relaxed isolates folding-backend effects.
- Decoy interpretation is strongest when decoy and reference have comparable processing history.

Debugging:

- Check that every case has the correct antigen and nanobody chain mapping.
- Confirm the reference manifests contain no user-home or stale paths.
- Confirm native-refold quality is computed against the intended reference state.

## Stage 3: Redesign Mode Definition

Purpose:
Define which antigen residues are allowed to change.

Modes:

- Hotspot: a small prioritized set of antigen residues near the nanobody interface.
- Interface: antigen residues at or near the nanobody-antigen interface.
- Full antigen: all antigen residues mutable; not part of the final validated package.

Inputs:

- Native complex coordinates.
- Antigen and nanobody chain IDs.
- Contact/interface calculation.

Outputs:

- Fixed-position masks.
- Redesign-mode manifests.
- ProteinMPNN-ready structure/chain inputs.

Design justification:

- Hotspot mode tests whether a small number of interface-proximal changes can perturb predicted binding while preserving fold.
- Interface mode tests a broader but still structurally focused redesign space.
- Full antigen is larger and more confounded by fold disruption, so it was not finalized for the current package.

Debugging:

- If too many residues are mutable, inspect contact-distance cutoffs and chain IDs.
- If too few residues are mutable, check whether antigen/nanobody chains were swapped or missing.
- A mutation profile should show interface-enriched changes for hotspot/interface modes.

## Stage 4: ProteinMPNN Sequence Generation

Purpose:
Generate plausible mutant antigen sequences within the allowed redesign mask.

Inputs:

- Native complex-derived structure inputs.
- Fixed-position/mutable-position masks.
- Chain metadata.
- ProteinMPNN runtime config.

Outputs:

- FASTA files and sequence manifests.
- Candidate sequence scores.

Design choice: 150 sequences per native

- 150 gives enough sequence diversity to sample different amino acid chemistries and mutation counts.
- It is still small enough to keep folding cost tractable.
- It provides a buffer against malformed candidates, duplicates, or candidates that later fail folding/QC.

Are mutations random?

- No. ProteinMPNN samples sequences from a learned conditional model given the structure and allowed mutable positions.
- The mutable positions are selected by Jango.
- The amino acids are sampled by ProteinMPNN according to model probabilities and temperature/settings.

Validation:

- Count FASTA headers.
- Confirm each target has the expected number of candidates.
- Confirm no empty or malformed FASTA files.
- Confirm mutation positions respect the allowed redesign mask.

Debugging:

- Empty FASTA usually indicates upstream structure parsing or chain/mask failure.
- Too few headers usually indicates interrupted ProteinMPNN generation.
- Mutations outside the mask indicate a fixed-position configuration problem.

## Stage 5: ProteinMPNN Filtering to Candidate Set

Purpose:
Reduce 150 sequence candidates to a foldable candidate set.

Current design:

- Select the best 5 sequence candidates per native/design mode.
- Use ProteinMPNN score/validity to choose plausible sequences.
- Remove malformed, duplicate, or incomplete candidates.

Why top 5?

- Folding every MPNN candidate is expensive.
- One candidate is too brittle because it over-trusts sequence scoring.
- Five candidates preserve modest diversity while keeping downstream folding practical.

Outputs:

- Fold input manifest.
- Candidate FASTA or sequence files for each selected design.

Validation:

- Expected rows equal selected candidates across all native cases.
- `fold_input` paths exist.
- MSA paths exist or are buildable if that mode is enabled.
- No stale paths such as user-home or retired tool roots.

Debugging:

- If later fold jobs fail with missing input paths, rerun the lineage audit before submitting.
- If top candidates appear over-mutated, inspect ProteinMPNN temperature and redesign mask size.

## Stage 6: Folding Backend Strategy

Purpose:
Convert mutant antigen sequences into 3D antigen structures.

Backends:

- ESM: canonical ESM folding backend used for final ESM results.
- OpenDDE: diffusion-based folding backend used for final OpenDDE results.

Shared input:

- Both backends consume the same MPNN-selected sequence set for a given redesign mode.

Design justification:

- Keeping MPNN input matched means backend comparisons mostly reflect folding/modeling behavior, not different sequence-generation pools.
- Running both backends allows detection of backend-specific fold distortion or overconfidence.

Validation:

- Confirm fold manifests have the same native/design IDs across backends.
- Confirm every row has existing input sequence and MSA path where needed.
- Confirm no stale path references.

## Stage 7: Fold Sampling and Top-1 Selection

Purpose:
Generate multiple structural hypotheses but retain one canonical decoy per native case.

Inputs:

- Fold input manifest.
- Backend-specific runtime config.
- Selected MPNN candidates.

Outputs:

- Raw folded antigen models.
- Fold provenance.
- Ranked top-1 fold table.
- Canonical top-1 folded decoys.

Why sampled folding?

- A single folded model may reflect one local structural hypothesis.
- Sampling captures model uncertainty and gives a chance to select a more plausible fold.
- OpenDDE natively uses diffusion-style sampling.
- ESM is used through the canonical sampled Jango backend configuration.

Why top-1?

- The final dataset must be balanced: one native case maps to one final decoy per backend/design mode.
- Keeping all samples would overweight cases with more successful intermediate models.
- Top-1 keeps the most plausible representative while retaining raw provenance elsewhere.

Ranking principle:

- Prefer backend-native confidence/provenance where available.
- Require valid structure, chain, residue, and sequence consistency.
- Do not use downstream Rosetta relax or DockQ to pick the fold; that would leak evaluation information into selection.

Validation:

- Every native case should have at most one canonical top-1 decoy per backend/design mode.
- Missing cases must be explicitly listed.
- Top-1 paths must point to canonical `data/folds` or run-root equivalents before promotion.

Debugging:

- If a backend finishes too quickly, inspect whether outputs are placeholders or actual PDBs.
- If PDB count is low, inspect per-shard logs and provenance.
- If top-1 selection is missing cases, check fold validity and chain-label consistency.

## Stage 8: Grafting

Purpose:
Place the folded mutant antigen back into nanobody complex context.

Inputs:

- Folded mutant antigen PDB.
- Native or reference complex scaffold.
- Chain mapping.
- Target antigen chain.

Outputs:

- Grafted decoy complex PDBs.
- Graft manifest.

How grafting works:

- The nanobody chain is kept from the reference complex.
- The antigen chain coordinates are replaced with the folded mutant antigen coordinates.
- Chain labels and residue ordering are reconciled.

Design justification:

- Folding backends produce antigen structures, but DockQ and Rosetta interface analysis require a complex.
- Keeping the nanobody fixed isolates antigen redesign effects.

Information loss:

- Side-chain and backbone details are inherited from the folded antigen and reference nanobody context.
- Any antigen-nanobody induced-fit effects are not modeled until relax.
- Grafting is inexpensive and can be regenerated, so pre-relax grafted PDBs are not core final outputs.

Validation:

- Every selected top-1 decoy has a grafted complex.
- Chain IDs match the manifest.
- Antigen residue count and sequence match the folded decoy.
- Nanobody chain is present and unchanged.

Debugging:

- Residue mismatch usually means chain label, residue numbering, or insertion-code handling failed.
- If grafted paths are stale, regenerate from canonical folded decoys and reference complexes rather than editing PDBs manually.

## Stage 9: Rosetta Relax

Purpose:
Remove steric artifacts and put decoy/reference complexes into a comparable Rosetta-scored state.

Inputs:

- Grafted decoy complexes.
- Native crystal/refold complexes where applicable.
- Rosetta runtime configuration.

Outputs:

- Relaxed decoy complexes.
- Relax manifests and status tables.
- Rosetta logs.

What gets relaxed?

- Grafted decoy complexes.
- Native refold complexes.
- Native crystal complexes when generating relaxed-native references.

Why Rosetta?

- Rosetta is a widely used structural refinement and interface-scoring toolkit.
- Relax reduces artifacts introduced by grafting and fold/reference mismatches.
- InterfaceAnalyzer provides consistent energy and interface metrics across decoys and references.

Design justification:

- Do not compare relaxed decoys to unrelaxed references unless the purpose is reference-bias analysis.
- Final decoy-vs-refold interpretation uses relaxed decoy and relaxed native refold for comparable processing.

Validation:

- Relaxed PDB count matches expected decoy count.
- Manifest statuses are successful or explicitly skipped-existing-valid.
- No `Traceback`, `ERROR`, `OOM`, `No such file`, or stale path strings in logs/manifests.

Debugging:

- If relax is interrupted, consolidate successful relaxed PDBs, inventory missing IDs, and rerun only incomplete cases.
- If Docker/Apptainer fails, check Rosetta runtime path in `jango doctor`.

## Stage 10: DockQ Scoring

Purpose:
Measure interface structural similarity between a decoy complex and a reference complex.

Inputs:

- Relaxed decoy complex.
- Reference complex for the selected reference state.
- Chain mapping.

Outputs:

- DockQ table.
- DockQ component metrics when needed for audits.

Why DockQ?

- DockQ is a composite structural interface metric.
- It captures native contact recovery, ligand RMSD, and interface RMSD.
- It is not a fold-confidence score and not an energy score.

Reference choices:

- Refold reference: primary final comparison because backend-folded decoys and backend-refolded natives have comparable folding history.
- Relaxed reference: tests Rosetta/reference processing effects.
- Crystal reference: experimental coordinate anchor for reference-benchmark analysis.

Validation:

- DockQ table has one row per expected decoy.
- Failed rows are explained and excluded only if unavoidable.
- Chain mapping must match decoy/reference complex labels.

Debugging:

- DockQ failures are often chain-map or missing-PDB problems.
- Rerun path audits before resubmitting score jobs.

## Stage 11: Rosetta Interface Analysis

Purpose:
Compute structural and energetic interface features.

Inputs:

- Relaxed decoy/reference complexes.
- Rosetta InterfaceAnalyzer config.

Outputs:

- Interface score tables.
- Shape-complementarity tables.
- Protein-interface feature tables.

Important metrics:

- Interface energy and separated interface energy in REU.
- Shape complementarity.
- Buried SASA.
- Hydrogen bond and contact-related features.

Design justification:

- Rosetta metrics help describe predicted interface changes.
- Rosetta energy is not calibrated to experimental affinity, so it is used as a relative computational coordinate.
- Public figure axis uses `decoy vs refold energy (REU)` rather than experimental delta-G language.

Validation:

- Table row count matches expected decoy/reference count.
- Required columns are present.
- No stale source paths remain.

Debugging:

- If InterfaceAnalyzer output is empty, inspect Rosetta log and input PDB chain IDs.
- If scores are extreme, check for missing chains, broken grafts, or relax failures.

## Stage 12: Native Refold Quality Analysis

Purpose:
Ask whether folding the native antigen changes its structure enough to confound decoy interpretation.

Inputs:

- Native crystal or relaxed reference antigen.
- Backend-refolded native antigen/complex.

Outputs:

- Native refold quality table.
- TM-score histograms.
- Backend comparison histograms.

Metrics:

- TM-score.
- Antigen RMSD where available.

Why TM-score?

- TM-score is length-normalized and more interpretable across antigens of different sizes than raw RMSD alone.
- A TM-score above 0.90 indicates high global structural similarity for this project.

Design justification:

- If a backend cannot refold native antigen accurately, decoy-vs-reference results are less trustworthy.
- Refold quality is attached to decoys in the ML-ready table so downstream users can filter by backend/reference reliability.

Validation:

- Native refold count matches expected cases minus documented failures.
- TM-score distributions are computed against the intended reference state.
- Backend labels are `ESM` and `OpenDDE`.

## Stage 13: Decoy Landscape Analysis

Purpose:
Summarize whether decoys are structurally preserved and computationally energy-shifted.

Inputs:

- DockQ table.
- Rosetta interface table.
- Native refold quality table.
- Decoy metadata.

Outputs:

- Decoy landscape figures.
- Redesign-mode comparison figures.
- Backend comparison figures.
- `decoy_landscape_ml_dataset.csv`.
- `decoy_figure_stats.csv`.

Canonical Q1 definition:

- DockQ >= 0.49.
- Decoy-vs-refold energy coordinate >= 0.

Design justification:

- DockQ provides structural/interface preservation.
- The energy coordinate provides the predicted Rosetta interface shift.
- Q1 is a practical quadrant for decoys that preserve interface structure while changing predicted interface energetics.

Validation:

- Total decoy count, Q1 count, median DockQ, median energy coordinate, and TM90-filtered Q1 counts are reported.
- Figures and tables use the same Q1 definition.

Debugging:

- If graph stats and table stats disagree, confirm they are generated from the same compact table.
- If outliers distort plots, keep data in tables but apply visual axis limits transparently.

## Stage 14: Mutation Profile Analysis

Purpose:
Identify mutation patterns associated with high-quality decoys.

Inputs:

- ML-ready decoy table.
- Native and mutant sequences.
- Redesign-mode metadata.
- Interface residue annotations.

Outputs:

- Mutation count map.
- Mutation chemistry enrichment figure.
- Mutation profile summary table.
- Mutation landscape decoy table.
- Mutation chemistry enrichment table.

Core questions:

- Do better decoys have fewer mutations?
- Which amino acid chemistry changes are enriched in high-quality decoys?
- How does BLOSUM62 change relate to decoy quality?

Design choices:

- Use Q1 + TM90 as the high-confidence subset.
- Compare against the remaining decoy background.
- Use BLOSUM62 labels for substitution plausibility.
- Avoid antigen residue-number analysis across unrelated antigens, because global numbering is not biologically aligned across diverse targets.

Validation:

- The mutation-count figure includes correlation statistics.
- The chemistry plot axis explicitly says BLOSUM62.
- Tables contain the data required to regenerate the figures.

Debugging:

- If mutation counts are impossible, inspect sequence alignment and allowed-position masks.
- If chemistry classes are sparse, aggregate categories rather than overplotting rare substitutions.

## Stage 15: Reference-Bias Analysis

Purpose:
Measure how much reference processing changes Rosetta energy interpretation.

Inputs:

- Crystal native Rosetta table.
- Relaxed native Rosetta table.
- Native-refold Rosetta table.
- Decoy Rosetta table.
- Q1 decoy annotations.

Outputs:

- Reference-bias boxplot/funnel figure.
- Backend bias-excess landscapes.
- Reference-bias summary tables.
- Bias-bin tables.

Bias types:

- Crystal-to-relaxed bias: effect of Rosetta relaxation on native reference energy.
- Crystal-to-refold bias: effect of folding/refolding plus relaxation relative to crystal.
- Relaxed-to-refold bias: effect of backend refolding after both structures have relaxed context.

Why per-point bias?

- Each native case can have a different reference shift.
- Dataset-wide averages hide case-specific reference instability.
- Per-point bias lets each decoy carry the uncertainty of its own native reference.

Bias-excess idea:

- A decoy is more reference-robust if its energy shift is larger than the reference shift for the matched native case.
- This shrinks the Q1 pool to cases less likely to be explained by reference processing alone.

Design justification:

- This protects against overclaiming decoy signal when folding/relaxing the native reference already moves Rosetta energy substantially.

Validation:

- Boxplots are split by backend where relevant.
- Funnel plot reports how many Q1 decoys survive the stricter reference filter.
- Bias-excess landscapes show retained and filtered points clearly.

Debugging:

- If bias seems larger than decoy signal, report that limitation rather than renormalizing away the effect.
- Do not convert Rosetta REU into experimental affinity.

## Stage 16: Figure and Table Publication

Purpose:
Create a compact, reviewable output package.

Inputs:

- Canonical decoy/reference tables.
- Mutation profile tables.
- Reference-bias tables.

Outputs:

- `data/outputs/evaluation/decoys/figures`
- `data/outputs/evaluation/decoys/tables`
- `data/outputs/evaluation/validation/figures`
- `data/outputs/evaluation/validation/tables`
- Flat handover copies under `handover/figures` and `handover/tables`.

Core figures:

- Decoy-vs-refold landscapes by backend and redesign mode.
- Backend comparison landscapes.
- Native refold quality histograms.
- Redesign-mode comparison landscapes.
- Mutation count maps.
- Mutation chemistry enrichment plots.
- Reference-bias boxplot/funnel.
- Bias-excess landscapes.

Design justification:

- Keep only figures that answer project questions.
- Remove component-only, exploratory, obsolete, or duplicate figures.
- Keep small tables that reproduce figures and preserve ML-ready records.

Validation:

- Exactly 21 PNGs in `data/outputs/evaluation`.
- No retired interface-density map.
- No stale public labels.
- Handover checksums match current files.

## Stage 17: Handover Package

Purpose:
Make the project reproducible for a newcomer.

Inputs:

- Final evaluation figures/tables.
- Source reference-benchmark tables.
- Protocol docs.
- Command templates.

Outputs:

- `<JANGO_REPO>/handover`

Contents:

- `README.md`: entry point.
- `FULL_SOP.md`: full project protocol and design rationale.
- `DATA_LOCATION_TREE.txt`: tree-style visual map of canonical data locations.
- `REPRODUCE_FIGURES.md`: figure regeneration instructions.
- `LOCATION_MAP.md`: where data live.
- `DATASET_MANIFESTS.md`: table/manifest roles.
- `VALIDATION_CHECKLIST.md`: final checks.
- `commands/`: runnable templates.
- `figures/`, `tables/`, `source/`, `maps/`, `checksums/`.

Validation command:

```bash
bash handover/commands/validate_handover.sh
```

## Stage 18: Final Validation Before Commit

Run:

```bash
cd <JANGO_REPO>
bash handover/commands/validate_handover.sh
git status --short
```

Expected:

- `handover validation passed`.
- Intended files are staged.
- No unrelated user scratch files are staged.

Commit guidance:

- Commit handover docs/package and any source files required by the validated commands.
- Keep unrelated dirty files out of the handover commit unless reviewed.

## Stage 19: Known Limitations

- Rosetta energy is a computational coordinate, not experimental affinity.
- Q1 is an operational quadrant, not a biological truth label.
- Native refold quality is backend- and reference-state dependent.
- Antigen residue numbering is not globally comparable across unrelated targets.
- Full-antigen redesign was not finalized in the current validated package.
- SKEMPI experimental validation remains future work.

## Stage 20: Future Work

- Add a formal user FASTA/CIF onboarding command.
- Add SKEMPI exact-mutant sequence generation.
- Add experimental correlation analysis for SKEMPI delta-delta-G.
- Improve calibration of Rosetta-derived energy coordinates.
- Expand per-stage troubleshooting examples as new users run the pipeline.
