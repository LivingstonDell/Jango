# Folding backend integration for DELPHI-VHH decoys

Historical note: these backend-integration files originally referred to `~/nanobody-interface-atlas-main`. In Jango, NBIA lives under `src/nbia`; treat this document as backend-reference material until the ESMFold2 adapter is re-documented.

```text
src/nbia/folding/__init__.py
src/nbia/folding/base.py
src/nbia/folding/boltz2_backend.py
src/nbia/folding/esmfold2_backend.py
scripts/nbia/run_esmfold2_monomer.py
```

Then update `src/nbia/decoys.py` and `src/nbia/cli.py` to call the backend interface.

The generated `decoys.py`/`cli.py` files from the previous step already implement a direct version of this. This backend package is the cleaner long-term layout:
- `decoys.py` should call `get_fold_backend(fold_backend)`
- backend writes inputs, job bundles, and discovers predictions
- validation/grafting/Rosetta remain backend-independent

Runtime requirements for ESMFold2:

```bash
pip install "esm@git+https://github.com/Biohub/esm.git@main"
export BIOHUB_TOKEN="..."
```

Expected new directories:

```text
work/decoy/<mode>/inputs/<structure>/esmfold2_monomer/
work/decoy/<mode>/esmfold2_monomer_outputs/
work/decoy/<mode>/job_bundle/esmfold2_monomer_jobs.tsv
work/decoy/<mode>/job_bundle/run_esmfold2_monomer_jobs.sh
```

