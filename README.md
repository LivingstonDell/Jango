# Jango

**Note:** This repository was developed during my work at the **Institute for Protein Innovation (IPI)**. The specific execution scripts, Slurm environment configurations, and hardcoded cluster paths reflect internal IPI infrastructure and are no longer actively deployed on a live environment. However, the core algorithmic logic, structural grafting pipelines, evaluation workflows, and orchestration principles remain fully intact and serve as a demonstration of the engineering framework.

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
| SAbDab2 input curation | Done | Curated shared inputs prepared for cluster storage. |
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

## Quick Start (Legacy HPC Setup)

> **Note on Environment:** The execution commands below rely on the environment module and Slurm cluster layout originally configured at IPI.

Users activate the Conda environment to load default environment variables:

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate jango
cd /path/to/jango_repo
