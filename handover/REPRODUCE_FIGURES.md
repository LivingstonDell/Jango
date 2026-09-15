# Reproduce Figures

The compact figure package is generated from the source tables bundled in this handover package. The final evaluation directory intentionally keeps only report-facing figures and compact tables.

## Environment

```bash
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango
jango doctor
```

## Regeneration flow

1. Generate the reference benchmark source package.
2. Generate mutation-profile source tables from the reference benchmark tables.
3. Publish the compact evaluation package.
4. Validate the final package.

The exact command templates are in `commands/`.

Bundled reference source tables live at:

```text
<JANGO_REPO>/handover/source/reference_benchmark
```

## Final expected outputs

```text
data/outputs/evaluation/decoys/figures      14 PNGs
data/outputs/evaluation/validation/figures   7 PNGs
data/outputs/evaluation                      21 PNGs total
```

Important public-label constraints:

- Public figures and tables use `ESM` as the backend label.
- Use `OpenDDE`, not ad hoc variants.
- Q1 is defined as DockQ >= 0.49 and delta dG >= 0.
