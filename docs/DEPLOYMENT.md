# Deployment

The intended shared deployment is:

```text
<JANGO_REPO>                 # clean Git source repository
<CONDA_ROOT>/envs/jango     # named Conda environment, separate from Git
```

Use the named environment for interactive checks:

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango
cd <JANGO_REPO>

jango doctor
fett preflight --input /path/to/raw/data --run-id validation_50 --max-structures 50
```

Normal runs use one per-user workspace:

```text
<JANGO_USER_ROOT>/work/<run-id>
<JANGO_USER_ROOT>/cache
<JANGO_USER_ROOT>/registry
<JANGO_USER_ROOT>/tmp
```

Do not create new top-level categories outside `cache`, `work`, `outputs`, and `tmp`; encode meaning in the run ID and stored metadata.

External runtimes such as ProteinMPNN, Boltz2, ESMFold2, Rosetta/Apptainer, AbForge/MSA caches, DockQ-RS, ANARCI, and Slurm are configured through administrator-maintained production env files and resolved by Fett. Lightweight verification must not launch scientific workloads.
