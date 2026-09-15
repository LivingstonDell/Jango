# Jango Hotspot Final Report

Date: 2026-08-17

Repository snapshot: `<JANGO_REPO>`

Current HEAD: `16cf6ca` (`Add reference benchmark and mutation profile analyses`, 2026-08-14)

Current working tree note: the repository has uncommitted README/Q1 cleanup changes, figure-command default-path cleanup changes, and the InterfaceAnalyzer CLI feature files. These should be reviewed and committed before GitHub publication.

## Executive Summary

Jango was built to generate antigen decoys that preserve nanobody/antigen complex geometry while worsening predicted binding. The validated hotspot workflow now supports:

- ProteinMPNN hotspot antigen redesign.
- ESM and OpenDDE folding backends.
- Fold top-1 selection.
- Grafting folded mutant antigens back into the nanobody complex.
- Rosetta relax/interface scoring.
- DockQ and TM-score based structural evaluation.
- Reference-bias analysis against crystal, relaxed, and refolded references.
- Mutation-profile analysis for OpenDDE hotspot decoys.

The core result is that both folding backends can generate many hotspot decoys in the desired Q1 region: high DockQ structural preservation and positive Rosetta delta dG. However, the strongest scientific caveat is that Rosetta delta dG is a model-derived score, not an experimentally calibrated affinity. Reference choice, folding, and relaxation introduce measurable energy bias, so Q1 decoys should be interpreted as computationally promising rather than experimentally validated affinity-weakening variants.

## Frozen Project State

Canonical shared outputs are under:

```text
<JANGO_REPO>/data/outputs
+-- mpnn/
+-- folds/
+-- relaxed/
+-- evaluation/
```

Approximate curated output sizes:

| Output area | Size | Purpose |
|---|---:|---|
| `data/mpnn` | 196 MB | ProteinMPNN sequences and redesign inputs by mode |
| `data/folds` | 12 GB | Folded antigen outputs by backend/mode |
| `data/relaxed` | 4.5 GB | Relaxed decoy/reference complexes |
| `data/outputs/evaluation` | 91 MB | Evaluation tables, figures, reports, and source tables |

The report-ready hotspot package is split into:

```text
<JANGO_REPO>/data/outputs/evaluation/decoys
<JANGO_REPO>/data/outputs/evaluation/validation
```

Together these contain the compact canonical decoy and validation figure package.

Core figure groups:

- `decoys/figures/hotspot/...`
- `decoys/figures/interface/...`
- `decoys/figures/native_refold/...`
- `decoys/figures/redesign_mode/...`
- `validation/figures/mutation_profile/...`
- `validation/figures/reference_bias/...`

Minimal tables:

- `decoys/tables/decoy_landscape_ml_dataset.csv`
- `decoys/tables/native_refold_quality.csv`
- `decoys/tables/decoy_figure_stats.csv`
- `validation/tables/mutation_landscape_decoys.csv`
- `validation/tables/mutation_chemistry_enrichment.csv`
- `validation/tables/mutation_landscape_correlation_summary.csv`
- `validation/tables/reference_bias_decoys.csv`
- `validation/tables/reference_bias_summary.csv`

Exploratory analysis add-ons remain available for review under:

```text
docs/analysis_addons
```

The canonical shared evaluation output is intentionally limited to `decoys/` and `validation/`.

## Scientific Question

The primary question was:

Can Jango produce antigen decoys that retain useful structural similarity to the native nanobody/antigen complex while perturbing predicted binding?

The operational definition used here is Q1:

- DockQ above the structural preservation cutoff.
- Positive Rosetta delta dG relative to the selected reference.

For the main hotspot analysis, decoys were compared against the corresponding relaxed native refold reference. Additional crystal and relaxed reference comparisons were generated to measure reference-dependent bias.

## Pipeline Summary

1. Start with curated SAbDab2 nanobody/antigen complexes.
2. Identify hotspot antigen residues for redesign.
3. Use ProteinMPNN to generate candidate mutant antigen sequences.
4. Fold mutant antigens with ESM or OpenDDE.
5. Select top-1 folded decoy per native/design group.
6. Graft the folded antigen into the native nanobody complex context.
7. Relax the grafted complex with Rosetta.
8. Score interface energy and shape/contact metrics.
9. Compare decoy complexes to native references with DockQ and reference metrics.
10. Analyze Q1 success, reference bias, and mutation patterns.

## Backend Fold Controls

Native refolds were used to ask whether each folding backend preserves the antigen structure before decoy mutations are interpreted.

| Backend | Median native refold TM-score |
|---|---:|
| ESM | 95.324% |
| OpenDDE | 96.295% |

Both backends had median native-refold TM-scores above 90%, supporting their use for this dataset. This does not mean every individual antigen refold is accurate; per-decoy native-refold TM-score is retained in the ML/core tables so downstream filters can require high-quality native refolding.

## Hotspot Decoy Results

Main hotspot Q1 results use the refold-relaxed reference comparison.

| Backend | n decoys | n Q1 | n Q1 + TM>=90 | Q1 median DockQ | Q1 median delta dG |
|---|---:|---:|---:|---:|---:|
| ESM | 700 | 521 | 434 | 0.687 | 11.233 |
| OpenDDE | 702 | 546 | 463 | 0.690 | 10.534 |

Both backends produced a large Q1 pool. OpenDDE had slightly more total hotspot Q1 decoys and Q1+TM90 decoys, while median DockQ and delta dG were very similar between backends.

When requiring TM>=90 within Q1:

| Backend | Q1+TM90 median DockQ | Q1+TM90 median delta dG |
|---|---:|---:|
| ESM | 0.697 | 12.408 |
| OpenDDE | 0.700 | 11.253 |

These values suggest that many high-refold-quality decoys still preserve interface-like geometry while showing positive Rosetta energy perturbation.

## Reference Bias Analysis

Reference-bias analysis asks how much the interpretation changes when a decoy is compared to crystal, relaxed, or refold-relaxed references.

The key issue is that folding and relaxation change Rosetta interface energies even for native/reference structures. Therefore, a decoy with positive delta dG relative to a refolded reference may not remain equally convincing when compared to the crystal or relaxed reference.

The curated package uses:

- `reference_bias_crystal_minus_refold`
- `reference_bias_relaxed_minus_refold`
- `reference_bias_relaxed_minus_crystal`
- bias bins based on `reference_bias_crystal_minus_refold`
- `bias_excess_delta_dg`, which compares each decoy's delta dG against the expected bias for similar native reference behavior

Q1 filtering shrinks when reference robustness is required:

| Backend | Original Q1 | Bias-excess Q1 >=2 | Reference-robust Q1 + TM>=90 |
|---|---:|---:|---:|
| ESM | 521 | 240 | 174 |
| OpenDDE | 546 | 225 | 181 |

This is important. The less conservative Q1 pool is useful for exploration, but the reference-robust Q1+TM90 pool is the more defensible set for claims about decoys that remain promising after controlling for folding/reference effects.

Core reference-bias figures:

![Q1 reference bias and funnel](analysis_addons/hotspot_reference_bias/figures/hotspot_q1_reference_bias_boxplot_funnel.png)

![ESM bias-excess landscape](analysis_addons/hotspot_reference_bias/figures/hotspot_esm_bias_excess_landscape.png)

![OpenDDE bias-excess landscape](analysis_addons/hotspot_reference_bias/figures/hotspot_opendde_bias_excess_landscape.png)

## Mutation Profile Analysis

The concise mutation-profile analysis currently focuses on OpenDDE hotspot decoys. It asks what mutation patterns are enriched among Q1+TM90 decoys.

OpenDDE hotspot mutation landscape:

| Quantity | Value |
|---|---:|
| Decoys profiled | 702 |
| Q1+TM90 decoys | 463 |
| Median mutations per decoy | 8 |
| Median mutations per Q1+TM90 decoy | 8 |
| Median interface mutation fraction | 0.75 |
| Median Q1+TM90 interface mutation fraction | 0.75 |
| Median mean BLOSUM62 | -0.20 |
| Median Q1+TM90 mean BLOSUM62 | -0.25 |

The chemistry enrichment table suggests that Q1+TM90 decoys are enriched for aromatic-involved mutations and modestly enriched for charge-changing mutations, while glycine/proline-involved mutations are depleted.

Top chemistry patterns:

| Feature | log2 odds ratio | Q1+TM90 frequency | Background frequency | Fisher p-value |
|---|---:|---:|---:|---:|
| Aromatic involved | 0.602 | 0.222 | 0.158 | 6.68e-10 |
| Charge changed | 0.165 | 0.429 | 0.401 | 0.0317 |
| Gly/pro involved | -0.300 | 0.172 | 0.204 | 0.00187 |

These results are descriptive, not causal. They identify patterns associated with the high-quality computational decoy pool, not experimentally proven rules.

Core mutation-profile figures:

![Hotspot ESM mutation count map](../data/outputs/evaluation/validation/figures/mutation_profile/esm/hotspot_mutation_count_map.png)

![Hotspot OpenDDE mutation count map](../data/outputs/evaluation/validation/figures/mutation_profile/opendde/hotspot_mutation_count_map.png)

![Hotspot ESM mutation chemistry enrichment](../data/outputs/evaluation/validation/figures/mutation_profile/esm/hotspot_mutation_chemistry_enrichment.png)

![Hotspot OpenDDE mutation chemistry enrichment](../data/outputs/evaluation/validation/figures/mutation_profile/opendde/hotspot_mutation_chemistry_enrichment.png)

## Interpretation

The hotspot workflow is complete and produces a meaningful computational decoy set. The strongest supported claims are:

- ESM and OpenDDE native refolds are generally structurally accurate by median TM-score.
- Both backends can produce many hotspot decoys with preserved DockQ and positive Rosetta delta dG.
- A stricter reference-robust filter substantially reduces the Q1 pool, which shows that reference/fold/relax bias matters.
- Mutation patterns in OpenDDE hotspot Q1+TM90 decoys suggest useful associations with aromatic involvement, charge change, and avoidance of gly/pro involvement.

The strongest limitations are:

- Rosetta delta dG is not a reliable experimental affinity measurement.
- The current Q1 interpretation is computational and should be validated against experimental data such as SKEMPI-2 where possible.
- Mutation-profile analysis is strongest for OpenDDE hotspot; it should not yet be generalized across all modes/backends.
- Full-antigen redesign was not completed in the final validated analysis.
- Interface-only has generated outputs, but this final report focuses on the hotspot package.

## Recommended Report/Handoff Assets

For a compact handoff, use:

```text
<JANGO_REPO>/data/outputs/evaluation/decoys
<JANGO_REPO>/data/outputs/evaluation/validation
```

For a complete reproducibility trace, also retain:

```text
<JANGO_REPO>/data/mpnn
<JANGO_REPO>/data/folds
<JANGO_REPO>/data/relaxed
```

For review of exploratory analyses that are report-ready but not yet first-class Jango commands, use:

```text
<JANGO_REPO>/docs/analysis_addons
```

The local figure mirror is:

```text
<LOCAL_JANGO_COPY>\data\outputs\evaluation\decoys
<LOCAL_JANGO_COPY>\data\outputs\evaluation\validation
```

## Next Steps

Before putting the project on IPI GitHub:

1. Commit the current code/docs cleanup and InterfaceAnalyzer changes if they pass review.
2. Update the GitHub-facing README with a progress checklist.
3. Add or verify `.gitignore` rules for work directories, tmp folders, Slurm logs, caches, and bulky generated data.
4. Decide whether curated small figures/tables stay in Git, while large PDB/fold/relax datasets remain on shared storage or release artifacts.
5. Add SKEMPI-2 validation as the next major scientific validation track.
6. Add custom CIF/PDB and custom FASTA manifest onboarding for external users.
7. Treat Rosetta dG as a relative computational feature unless and until experimental validation supports a calibrated metric.
