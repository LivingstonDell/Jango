# Jango Location Map

## Primary locations

| Location | Role | Keep? | Notes |
|---|---|---:|---|
| `<JANGO_REPO>` | Canonical shared Jango repo on <SLURM_NODE> | Yes | Source code, shared configs, canonical data outputs. |
| `<JANGO_REPO>/data/folds` | Canonical folded monomer outputs | Yes | Split by backend and redesign mode. |
| `<JANGO_REPO>/data/mpnn` | Canonical ProteinMPNN sequence outputs | Yes | Split by redesign mode. |
| `<JANGO_REPO>/data/relaxed` | Canonical relaxed complex outputs | Yes | Top-1 relaxed decoys and native refolds. |
| `<JANGO_REPO>/data/outputs/evaluation` | Canonical compact report-facing figures/tables | Yes | Final package: decoys + validation. |
| `<JANGO_REPO>/handover/source/reference_benchmark` | Bundled source CSVs for handover figure reproduction | Yes | Lightweight source tables copied from the completed benchmark run. |
| `<JANGO_REPO>/data/msa` | MSA cache root used by Jango/AbForge helper | Yes | Do not rename casually; jobs depend on this. |
| `<JANGO_REPO>/data/msa_cache` | Separate MSA cache compatibility/cache layer | Yes | Keep both MSA roots unless deliberately migrated. |
| `<JANGO_USER_ROOT>` | User work area on <SLURM_NODE> | No | Temporary runs and disposable work only; canonical handover should not depend on this. |
| `<LOCAL_JANGO_COPY>` | Local mirror of figures/tables/docs | Yes | Convenience mirror, not the source of truth for data. |

## Canonical evaluation layout

```text
<JANGO_REPO>/data/outputs/evaluation
├── decoys
│   ├── figures
│   └── tables
└── validation
    ├── figures
    └── tables
```

## Final heavy data roots

```text
<JANGO_REPO>/data/folds/esmfold2/{hotspot_only,interface_only}
<JANGO_REPO>/data/folds/opendde/{hotspot_only,interface_only}
<JANGO_REPO>/data/mpnn/{hotspot_only,interface_only,full_antigen}
<JANGO_REPO>/data/relaxed/esmfold2_hotspot_top1
<JANGO_REPO>/data/relaxed/esmfold2_interface_only
<JANGO_REPO>/data/relaxed/esmfold2_native_refold
<JANGO_REPO>/data/relaxed/opendde_hotspot_top1
<JANGO_REPO>/data/relaxed/opendde_interface_only
<JANGO_REPO>/data/relaxed/opendde_native_refold
```
