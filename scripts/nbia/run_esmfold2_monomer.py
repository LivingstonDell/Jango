#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ESMFOLD2_MODEL = "Biohub/ESMFold2"
DEFAULT_ESMFOLD2_DEVICE = "cuda"


def env_value(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        value = default
    if required and (value is None or value == ""):
        raise SystemExit(
            f"Required environment variable is not set: {name}. "
            "Set it or pass the corresponding command-line argument."
        )
    return value


def env_path(name: str, default: str | Path | None = None, required: bool = False) -> Path | None:
    value = env_value(name, str(default) if default is not None else None, required=required)
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def read_fasta(path: Path) -> tuple[str, str]:
    name = path.stem
    seq_parts: list[str] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0]
        else:
            seq_parts.append(line)

    sequence = "".join(seq_parts).upper()
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")

    return name, sequence



def normalize_msa_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper().replace("-", "").replace(".", "")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(normalize_msa_sequence(sequence).encode("utf-8")).hexdigest()


def load_a3m_msa(msa_path: Path, query_sequence: str, max_sequences: int | None = None):
    """Load and validate an A3M with the installed ESMFold2 MSA implementation."""

    from esm.utils.msa import MSA

    if not msa_path.is_file():
        raise FileNotFoundError(f"MSA A3M file does not exist: {msa_path}")
    try:
        msa = MSA.from_a3m(msa_path, remove_insertions=True, max_sequences=max_sequences)
    except Exception as exc:
        raise ValueError(f"Unable to parse A3M with ESMFold2 MSA.from_a3m: {msa_path}: {exc}") from exc
    if msa.depth <= 0:
        raise ValueError(f"MSA contains no sequences: {msa_path}")
    query = normalize_msa_sequence(query_sequence)
    msa_query = normalize_msa_sequence(msa.query)
    if msa_query != query:
        raise ValueError(
            "A3M query sequence does not match FASTA sequence: "
            f"query_hash={sequence_sha256(query_sequence)} msa_query_hash={sequence_sha256(msa.query)} "
            f"path={msa_path}"
        )
    return msa, {
        "msa_path": str(msa_path),
        "msa_format": "a3m",
        "msa_depth": int(msa.depth),
        "msa_query_length": int(msa.seqlen),
        "msa_query_sequence_hash": sequence_sha256(query_sequence),
        "msa_model_consumed": True,
    }


def build_esmfold2_input(sequence: str, msa):
    """Build the ESMFold2 StructurePredictionInput that carries MSA data."""

    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence, msa=msa)])


def convert_mmcif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    """Best-effort mmCIF to PDB conversion for downstream PDB-only validation."""

    try:
        import biotite.structure.io.pdb as pdb
        import biotite.structure.io.pdbx as pdbx

        cif_file = pdbx.CIFFile.read(str(cif_path))
        atoms = pdbx.get_structure(cif_file, model=1)
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(atoms)
        pdb_file.write(str(pdb_path))
        return pdb_path.exists()
    except Exception:
        return False


def to_numpy(x: Any):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def result_metric(result: Any, name: str):
    value = None
    if isinstance(result, dict):
        value = result.get(name)
    else:
        value = getattr(result, name, None)
    arr = to_numpy(value)
    if arr is None:
        return None
    try:
        return float(np.asarray(arr).mean())
    except Exception:
        return str(value)


def sample_sort_key(result: Any, index: int) -> tuple[float, float, float, int]:
    ptm = result_metric(result, "ptm")
    mean_plddt = result_metric(result, "plddt")
    mean_pae = result_metric(result, "pae")
    return (
        -(ptm if isinstance(ptm, float) else float("-inf")),
        -(mean_plddt if isinstance(mean_plddt, float) else float("-inf")),
        mean_pae if isinstance(mean_pae, float) else float("inf"),
        index,
    )


def select_best_result(results: list[Any]) -> tuple[Any, int]:
    if not results:
        raise ValueError("ESMFold2 returned no diffusion samples")
    ranked = sorted(enumerate(results), key=lambda item: sample_sort_key(item[1], item[0]))
    best_index, best_result = ranked[0]
    return best_result, best_index


def write_text_if_not_none(path: Path, value: Any) -> bool:
    if value is None:
        return False
    path.write_text(str(value))
    return True


def maybe_write_structure(result: Any, out_dir: Path, name: str) -> tuple[Path | None, Path | None]:
    cif_path = out_dir / f"{name}.cif"
    pdb_path = out_dir / f"{name}.pdb"

    cif_written = False
    pdb_written = False

    candidates = [
        result,
        getattr(result, "complex", None),
        getattr(result, "structure", None),
        getattr(result, "protein", None),
    ]

    for obj in candidates:
        if obj is None:
            continue

        if not cif_written:
            if hasattr(obj, "to_mmcif"):
                cif_path.write_text(obj.to_mmcif())
                cif_written = True
            elif hasattr(obj, "to_mmcif_string"):
                cif_path.write_text(obj.to_mmcif_string())
                cif_written = True
            elif hasattr(obj, "to_cif"):
                cif_path.write_text(obj.to_cif())
                cif_written = True

        if not pdb_written:
            if hasattr(obj, "to_pdb_string"):
                pdb_path.write_text(obj.to_pdb_string())
                pdb_written = True
            elif hasattr(obj, "to_pdb"):
                try:
                    value = obj.to_pdb()
                except TypeError:
                    obj.to_pdb(pdb_path)
                    pdb_written = pdb_path.exists()
                else:
                    if value is not None:
                        pdb_path.write_text(str(value))
                        pdb_written = True

    if cif_written and not pdb_written:
        pdb_written = convert_mmcif_to_pdb(cif_path, pdb_path)

    return (cif_path if cif_written else None, pdb_path if pdb_written else None)


def esmc_snapshot_path(cache_dir: Path) -> Path:
    return (
        Path(os.environ["ESMC_MODEL_PATH"]).expanduser().resolve()
        if os.environ.get("ESMC_MODEL_PATH")
        else cache_dir
        / "hub"
        / "models--biohub--ESMC-6B"
        / "snapshots"
        / "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
    )


class LocalESMFold2Session:
    """Reusable local ESMFold2 model session for sequential fold batches."""

    def __init__(self, *, model_id_or_path: str, cache_dir: Path, device: str) -> None:
        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        self.torch = torch
        self.model_id_or_path = model_id_or_path
        self.cache_dir = cache_dir
        self.device = device
        self.model = ESMFold2Model.from_pretrained(
            model_id_or_path,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            load_esmc=False,
        )
        self.model.load_esmc(str(esmc_snapshot_path(cache_dir)))
        self.model.eval()
        self.model.to(device)

    def fold(
        self,
        *,
        sequence: str,
        num_recycles: int,
        msa_a3m: Path | None = None,
        msa_max_sequences: int | None = None,
        num_sampling_steps: int | None = None,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
    ) -> tuple[Any, str | None, dict[str, object]]:
        if msa_a3m is not None:
            from esm.models.esmfold2.processor import ESMFold2InputBuilder

            msa, msa_metadata = load_a3m_msa(msa_a3m, sequence, max_sequences=msa_max_sequences)
            structure_input = build_esmfold2_input(sequence, msa)
            builder = ESMFold2InputBuilder()
            result = builder.fold(
                self.model,
                structure_input,
                num_loops=num_recycles,
                num_sampling_steps=num_sampling_steps or 200,
                num_diffusion_samples=num_diffusion_samples,
                seed=seed,
                complex_id="pred",
            )
            return result, None, msa_metadata

        if num_diffusion_samples != 1 or num_sampling_steps is not None or seed is not None:
            raise ValueError(
                "ESMFold2 sampling requires the MSA-backed ESMFold2InputBuilder path"
            )

        with self.torch.no_grad():
            try:
                result = self.model.infer_protein(sequence, num_loops=num_recycles)
            except TypeError:
                result = self.model.infer_protein(sequence)

            try:
                pdb_text = self.model.infer_protein_as_pdb(sequence, num_loops=num_recycles)
            except TypeError:
                pdb_text = self.model.infer_protein_as_pdb(sequence)

        return result, pdb_text, {"msa_model_consumed": False}

    def clear_memory(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "empty_cache") and str(self.device).startswith("cuda"):
            cuda.empty_cache()


def run_local_esmfold2(
    *,
    sequence: str,
    model_id_or_path: str,
    cache_dir: Path,
    device: str,
    num_recycles: int,
    msa_a3m: Path | None = None,
    msa_max_sequences: int | None = None,
    num_sampling_steps: int | None = None,
    num_diffusion_samples: int = 1,
    seed: int | None = None,
) -> tuple[Any, str | None, dict[str, object]]:
    """Run local ESMFold2 using supported inference APIs."""

    session = LocalESMFold2Session(
        model_id_or_path=model_id_or_path,
        cache_dir=cache_dir,
        device=device,
    )
    return session.fold(
        sequence=sequence,
        num_recycles=num_recycles,
        msa_a3m=msa_a3m,
        msa_max_sequences=msa_max_sequences,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
    )


def write_prediction_outputs(
    *,
    name: str,
    sequence: str,
    result: Any,
    pdb_text: str | None,
    msa_metadata: dict[str, object],
    out_dir: Path,
    model_id_or_path: str,
    device: str,
    cache_dir: Path,
    sample_index: int | None = None,
    num_diffusion_samples: int = 1,
    selected_sample_index: int | None = None,
    num_sampling_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / f"{name}.pdb"
    cif_path = None
    if pdb_text is not None:
        pdb_path.write_text(pdb_text)
    else:
        cif_path, pdb_path = maybe_write_structure(result, out_dir, name)

    metrics: dict[str, object] = {
        "name": name,
        "backend": "esmfold2",
        "execution_mode": "local",
        "model_id_or_path": model_id_or_path,
        "sequence_length": len(sequence),
        "device": device,
        "cache_dir": str(cache_dir),
        "num_diffusion_samples": int(num_diffusion_samples),
        "sample_index": sample_index,
        "selected_sample_index": selected_sample_index,
        "num_sampling_steps": num_sampling_steps,
        "seed": seed,
        "cif_path": str(cif_path) if cif_path else None,
        "pdb_path": str(pdb_path) if pdb_path else None,
        "wrote_cif": cif_path is not None,
        "wrote_pdb": pdb_path is not None,
        **msa_metadata,
    }

    if isinstance(result, dict):
        best_idx = None
        if "ptm" in result:
            ptm_arr = to_numpy(result.get("ptm"))
            if ptm_arr is not None:
                metrics["ptm"] = float(np.mean(ptm_arr))
                best_idx = int(np.nanargmax(ptm_arr))
                metrics["best_sample_index"] = best_idx

        if "iptm" in result:
            arr = to_numpy(result.get("iptm"))
            if arr is not None:
                metrics["iptm"] = float(np.mean(arr))

        for src_key, out_key in [
            ("plddt", "mean_plddt"),
            ("plddt_ca", "mean_plddt_ca"),
            ("pae", "mean_pae"),
            ("pde", "mean_pde"),
        ]:
            arr = to_numpy(result.get(src_key))
            if arr is None:
                continue
            np.save(out_dir / f"{name}_{src_key}.npy", arr)
            metrics[out_key] = float(np.mean(arr))
            if best_idx is not None and arr.shape[0] > best_idx:
                metrics[f"{out_key}_best_sample"] = float(np.mean(arr[best_idx]))
    else:
        for src_key, out_key in [
            ("ptm", "ptm"),
            ("iptm", "iptm"),
            ("plddt", "mean_plddt"),
            ("pae", "mean_pae"),
        ]:
            value = getattr(result, src_key, None)
            arr = to_numpy(value)
            if arr is None:
                continue

            if src_key in {"plddt", "pae"}:
                np.save(out_dir / f"{name}_{src_key}.npy", arr)
                metrics[out_key] = float(np.mean(arr))
            else:
                try:
                    metrics[out_key] = float(np.asarray(arr).mean())
                except Exception:
                    metrics[out_key] = str(value)

    (out_dir / f"{name}_confidence.json").write_text(json.dumps(metrics, indent=2))

    if pdb_path is None and cif_path is None:
        raise RuntimeError(
            "Local ESMFold2 ran but no structure writer was discovered on the result. "
            "Inspect the result object and update maybe_write_structure()."
        )

    if pdb_path is None:
        raise RuntimeError(
            "Local ESMFold2 wrote mmCIF but not PDB. Downstream grafting currently "
            "expects PDB, so add a CIF-to-PDB conversion step or update the local writer."
        )
    return metrics

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fold one protein monomer using local ESMFold2. No Biohub API."
    )
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--model-id-or-path", default=env_value("ESMFOLD2_MODEL", DEFAULT_ESMFOLD2_MODEL))
    p.add_argument("--cache-dir", type=Path, default=env_path("ESMFOLD2_CACHE"))
    p.add_argument("--esm-root", type=Path, default=env_path("ESM_ROOT"))
    p.add_argument("--device", default=env_value("ESMFOLD2_DEVICE", DEFAULT_ESMFOLD2_DEVICE))
    p.add_argument("--num-recycles", type=int, default=3)
    p.add_argument("--num-sampling-steps", type=int, default=None, help="ESMFold2 sampling steps for MSA-backed folding.")
    p.add_argument("--num-diffusion-samples", type=int, default=1, help="Number of ESMFold2 samples to generate per input. Default preserves single-sample behavior.")
    p.add_argument("--seed", type=int, default=None, help="Optional ESMFold2 sampling seed.")
    p.add_argument("--msa-a3m", type=Path, default=None, help="Validated unpaired A3M MSA passed into ESMFold2InputBuilder.")
    p.add_argument("--msa-max-sequences", type=int, default=None, help="Optional maximum MSA depth consumed from --msa-a3m.")
    p.add_argument("--require-msa", action="store_true", help="Fail if --msa-a3m is absent or unusable; never fall back to sequence-only mode.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.fasta = args.fasta.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.cache_dir is None:
        raise SystemExit("Required environment variable is not set: ESMFOLD2_CACHE. Set it or pass --cache-dir.")
    if args.esm_root is None:
        raise SystemExit("Required environment variable is not set: ESM_ROOT. Set it or pass --esm-root.")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.esm_root = args.esm_root.expanduser().resolve()
    args.msa_a3m = args.msa_a3m.expanduser().resolve() if args.msa_a3m is not None else None
    if args.num_diffusion_samples < 1:
        raise SystemExit("--num-diffusion-samples must be >= 1")
    if args.num_sampling_steps is not None and args.num_sampling_steps < 1:
        raise SystemExit("--num-sampling-steps must be >= 1")
    if args.require_msa and args.msa_a3m is None:
        raise SystemExit("--require-msa was set but --msa-a3m was not provided")
    if args.msa_a3m is not None and not args.msa_a3m.is_file():
        raise SystemExit(f"MSA A3M file does not exist: {args.msa_a3m}")

    os.environ.setdefault("ESM_ROOT", str(args.esm_root))
    os.environ.setdefault("ESMFOLD2_CACHE", str(args.cache_dir))
    os.environ.setdefault("HF_HOME", str(args.cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(args.cache_dir / "torch"))

    if str(args.esm_root) not in sys.path:
        sys.path.insert(0, str(args.esm_root))

    name, sequence = read_fasta(args.fasta)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "torch").mkdir(parents=True, exist_ok=True)

    result, pdb_text, msa_metadata = run_local_esmfold2(
        sequence=sequence,
        model_id_or_path=args.model_id_or_path,
        cache_dir=args.cache_dir,
        device=args.device,
        num_recycles=args.num_recycles,
        msa_a3m=args.msa_a3m,
        msa_max_sequences=args.msa_max_sequences,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=args.seed,
    )
    if isinstance(result, list):
        best_result, best_sample_index = select_best_result(result)
        for sample_index, sample_result in enumerate(result):
            sample_name = f"{name}_sample{sample_index:02d}"
            write_prediction_outputs(
                name=sample_name,
                sequence=sequence,
                result=sample_result,
                pdb_text=None,
                msa_metadata=msa_metadata,
                out_dir=args.out_dir / "samples",
                model_id_or_path=args.model_id_or_path,
                device=args.device,
                cache_dir=args.cache_dir,
                sample_index=sample_index,
                num_diffusion_samples=args.num_diffusion_samples,
                selected_sample_index=best_sample_index,
                num_sampling_steps=args.num_sampling_steps,
                seed=args.seed,
            )
        write_prediction_outputs(
            name=name,
            sequence=sequence,
            result=best_result,
            pdb_text=None,
            msa_metadata=msa_metadata,
            out_dir=args.out_dir,
            model_id_or_path=args.model_id_or_path,
            device=args.device,
            cache_dir=args.cache_dir,
            sample_index=best_sample_index,
            num_diffusion_samples=args.num_diffusion_samples,
            selected_sample_index=best_sample_index,
            num_sampling_steps=args.num_sampling_steps,
            seed=args.seed,
        )
    else:
        write_prediction_outputs(
            name=name,
            sequence=sequence,
            result=result,
            pdb_text=pdb_text,
            msa_metadata=msa_metadata,
            out_dir=args.out_dir,
            model_id_or_path=args.model_id_or_path,
            device=args.device,
            cache_dir=args.cache_dir,
            num_diffusion_samples=args.num_diffusion_samples,
            num_sampling_steps=args.num_sampling_steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
