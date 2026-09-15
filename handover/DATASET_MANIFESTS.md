# Dataset Manifests And Tables

## Minimal public tables

| Table | Role |
|---|---|
| `decoy_landscape_ml_dataset.csv` | One row per final decoy used for DockQ/delta-dG landscapes. |
| `native_refold_quality.csv` | Native refold TM-score data used for refold quality histograms. |
| `decoy_figure_stats.csv` | Summary stats for decoy and refold figures. |
| `mutation_landscape_decoys.csv` | Per-decoy mutation counts and chemistry features. |
| `mutation_chemistry_enrichment.csv` | Chemistry enrichment source table, including BLOSUM62-derived categories. |
| `mutation_landscape_correlation_summary.csv` | Statistical support for mutation/count relationships. |
| `reference_bias_decoys.csv` | Per-decoy reference-bias and bias-excess fields. |
| `reference_bias_summary.csv` | Reference-bias filter/funnel summary. |
| `reference_bias_bins.csv` | Expected delta dG by reference-bias bin. |

## Key interpretation fields

| Field | Meaning |
|---|---|
| `dockq` | Interface structural similarity score. |
| `delta_dg` | Rosetta dG shift relative to the selected reference state. |
| `high_high` | Q1 flag for DockQ >= 0.49 and delta dG >= 0. |
| `native_refold_tm_percent` | TM-score of the associated native refold, percent scale. |
| `native_refold_quality_pass` | Native refold TM >= 90%. |
| `n_residues_mutated` | Number of antigen sequence changes. |
| `mean_blosum62` | Mean BLOSUM62 substitution score across mutations. |
| `bias_excess_delta_dg` | Decoy delta dG beyond expected reference-bias bin effect. |

## ML-ready dataset

The most useful single table for ML-style downstream work is:

```text
<JANGO_REPO>/data/outputs/evaluation/decoys/tables/decoy_landscape_ml_dataset.csv
```

It should be joined with validation tables only when mutation chemistry or reference-bias fields are needed.
