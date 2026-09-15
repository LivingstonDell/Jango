
import contextlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "nbia" / "run_esmfold2_monomer.py"
JOBS_RUNNER = ROOT / "scripts" / "nbia" / "run_esmfold2_monomer_jobs.py"
TEST_TMP_ROOT = Path(os.environ.get("JANGO_TEST_TMP", Path(tempfile.gettempdir()) / "jango_test_tmp" / "esmfold2_msa_runner"))


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_esmfold2_monomer_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ESMFold2MSARunnerTests(unittest.TestCase):
    def test_help_exposes_msa_a3m(self):
        env = os.environ.copy()
        env.setdefault("ESMFOLD2_CACHE", str(TEST_TMP_ROOT / "esm_cache"))
        env.setdefault("ESM_ROOT", "/path/to/esmfold2")
        result = subprocess.run([sys.executable, str(RUNNER), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--msa-a3m", result.stdout)
        self.assertIn("--require-msa", result.stdout)

    def test_jobs_runner_help_exposes_batch_options(self):
        env = os.environ.copy()
        env.setdefault("ESMFOLD2_CACHE", str(TEST_TMP_ROOT / "esm_cache"))
        env.setdefault("ESM_ROOT", "/path/to/esmfold2")
        result = subprocess.run([sys.executable, str(JOBS_RUNNER), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--jobs-tsv", result.stdout)
        self.assertIn("--status-csv", result.stdout)

    def test_a3m_loader_uses_esm_msa_and_validates_query(self):
        module = load_runner_module()
        root = TEST_TMP_ROOT / "esm_msa_loader"
        root.mkdir(parents=True, exist_ok=True)
        msa_path = root / "query.a3m"
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")

        class FakeMSA:
            depth = 2
            seqlen = 4
            query = "ACDE"

            @classmethod
            def from_a3m(cls, path, remove_insertions=True, max_sequences=None):
                FakeMSA.called_with = (Path(path), remove_insertions, max_sequences)
                return cls()

        esm_mod = types.ModuleType("esm")
        utils_mod = types.ModuleType("esm.utils")
        msa_mod = types.ModuleType("esm.utils.msa")
        msa_mod.MSA = FakeMSA
        old = {name: sys.modules.get(name) for name in ["esm", "esm.utils", "esm.utils.msa"]}
        sys.modules.update({"esm": esm_mod, "esm.utils": utils_mod, "esm.utils.msa": msa_mod})
        try:
            msa, meta = module.load_a3m_msa(msa_path, "ACDE", max_sequences=7)
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertIsInstance(msa, FakeMSA)
        self.assertEqual(FakeMSA.called_with, (msa_path, True, 7))
        self.assertTrue(meta["msa_model_consumed"])
        self.assertEqual(meta["msa_depth"], 2)

    def test_reusable_session_loads_model_once_for_multiple_folds(self):
        module = load_runner_module()
        root = TEST_TMP_ROOT / "esm_session_reuse"
        root.mkdir(parents=True, exist_ok=True)

        class FakeNoGrad:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False

        class FakeCuda:
            calls = 0
            @classmethod
            def empty_cache(cls): cls.calls += 1

        torch_mod = types.ModuleType("torch")
        torch_mod.no_grad = lambda: FakeNoGrad()
        torch_mod.cuda = FakeCuda

        class FakeModel:
            loads = 0
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.loads += 1
                return cls()
            def load_esmc(self, path): self.esmc_path = path
            def eval(self): return self
            def to(self, device): self.device = device; return self
            def infer_protein(self, sequence, *args, **kwargs): return {"ptm": [0.5]}
            def infer_protein_as_pdb(self, sequence, *args, **kwargs): return "ATOM\nEND\n"

        modeling_mod = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
        modeling_mod.ESMFold2Model = FakeModel
        old = dict(sys.modules)
        sys.modules.update({
            "torch": torch_mod,
            "transformers.models.esmfold2.modeling_esmfold2": modeling_mod,
        })
        try:
            session = module.LocalESMFold2Session(model_id_or_path="fixture", cache_dir=root, device="cuda")
            session.fold(sequence="ACDE", num_recycles=3)
            session.clear_memory()
            session.fold(sequence="ACDF", num_recycles=3)
            session.clear_memory()
        finally:
            sys.modules.clear(); sys.modules.update(old)
        self.assertEqual(FakeModel.loads, 1)
        self.assertEqual(FakeCuda.calls, 2)

    def test_msa_reaches_esmfold2_input_builder_fold_call(self):
        module = load_runner_module()
        root = TEST_TMP_ROOT / "esm_msa_builder"
        root.mkdir(parents=True, exist_ok=True)
        msa_path = root / "query.a3m"
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")

        class FakeNoGrad:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False

        torch_mod = types.ModuleType("torch")
        torch_mod.no_grad = lambda: FakeNoGrad()

        class FakeModel:
            device = "cpu"
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def load_esmc(self, path): self.esmc_path = path
            def eval(self): return self
            def to(self, device): self.device = device; return self

        modeling_mod = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
        modeling_mod.ESMFold2Model = FakeModel

        class FakeMSA:
            depth = 2
            seqlen = 4
            query = "ACDE"
            @classmethod
            def from_a3m(cls, *args, **kwargs): return cls()

        class ProteinInput:
            def __init__(self, id, sequence, msa=None):
                self.id = id; self.sequence = sequence; self.msa = msa

        class StructurePredictionInput:
            def __init__(self, sequences): self.sequences = sequences

        input_builder_mod = types.ModuleType("esm.utils.structure.input_builder")
        input_builder_mod.ProteinInput = ProteinInput
        input_builder_mod.StructurePredictionInput = StructurePredictionInput

        class FakeResult:
            def to_mmcif(self): return "data_fake\n"
            def to_pdb_string(self): return "ATOM\nEND\n"

        class FakeBuilder:
            def fold(self, model, input, **kwargs):
                FakeBuilder.seen_input = input
                return FakeResult()

        processor_mod = types.ModuleType("esm.models.esmfold2.processor")
        processor_mod.ESMFold2InputBuilder = FakeBuilder
        msa_mod = types.ModuleType("esm.utils.msa")
        msa_mod.MSA = FakeMSA

        old = dict(sys.modules)
        sys.modules.update({
            "torch": torch_mod,
            "transformers.models.esmfold2.modeling_esmfold2": modeling_mod,
            "esm": types.ModuleType("esm"),
            "esm.models": types.ModuleType("esm.models"),
            "esm.models.esmfold2": types.ModuleType("esm.models.esmfold2"),
            "esm.models.esmfold2.processor": processor_mod,
            "esm.utils": types.ModuleType("esm.utils"),
            "esm.utils.msa": msa_mod,
            "esm.utils.structure": types.ModuleType("esm.utils.structure"),
            "esm.utils.structure.input_builder": input_builder_mod,
        })
        try:
            _, _, meta = module.run_local_esmfold2(
                sequence="ACDE",
                model_id_or_path="fixture-model",
                cache_dir=root,
                device="cpu",
                num_recycles=3,
                msa_a3m=msa_path,
            )
        finally:
            sys.modules.clear(); sys.modules.update(old)
        self.assertTrue(meta["msa_model_consumed"])
        self.assertIsInstance(FakeBuilder.seen_input.sequences[0].msa, FakeMSA)

    def test_msa_diffusion_options_reach_builder(self):
        module = load_runner_module()
        root = TEST_TMP_ROOT / "esm_msa_diffusion_options"
        root.mkdir(parents=True, exist_ok=True)
        msa_path = root / "query.a3m"
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")

        class FakeNoGrad:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False

        torch_mod = types.ModuleType("torch")
        torch_mod.no_grad = lambda: FakeNoGrad()

        class FakeModel:
            device = "cpu"
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def load_esmc(self, path): self.esmc_path = path
            def eval(self): return self
            def to(self, device): self.device = device; return self

        modeling_mod = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
        modeling_mod.ESMFold2Model = FakeModel

        class FakeMSA:
            depth = 2
            seqlen = 4
            query = "ACDE"
            @classmethod
            def from_a3m(cls, *args, **kwargs): return cls()

        class ProteinInput:
            def __init__(self, id, sequence, msa=None):
                self.id = id; self.sequence = sequence; self.msa = msa

        class StructurePredictionInput:
            def __init__(self, sequences): self.sequences = sequences

        input_builder_mod = types.ModuleType("esm.utils.structure.input_builder")
        input_builder_mod.ProteinInput = ProteinInput
        input_builder_mod.StructurePredictionInput = StructurePredictionInput

        class FakeResult:
            ptm = 0.7
            plddt = [0.8, 0.9]
            pae = [2.0, 3.0]
            def to_pdb_string(self): return "ATOM\nEND\n"

        class FakeBuilder:
            seen_kwargs = None
            def fold(self, model, input, **kwargs):
                FakeBuilder.seen_kwargs = kwargs
                return [FakeResult(), FakeResult()]

        processor_mod = types.ModuleType("esm.models.esmfold2.processor")
        processor_mod.ESMFold2InputBuilder = FakeBuilder
        msa_mod = types.ModuleType("esm.utils.msa")
        msa_mod.MSA = FakeMSA

        old = dict(sys.modules)
        sys.modules.update({
            "torch": torch_mod,
            "transformers.models.esmfold2.modeling_esmfold2": modeling_mod,
            "esm": types.ModuleType("esm"),
            "esm.models": types.ModuleType("esm.models"),
            "esm.models.esmfold2": types.ModuleType("esm.models.esmfold2"),
            "esm.models.esmfold2.processor": processor_mod,
            "esm.utils": types.ModuleType("esm.utils"),
            "esm.utils.msa": msa_mod,
            "esm.utils.structure": types.ModuleType("esm.utils.structure"),
            "esm.utils.structure.input_builder": input_builder_mod,
        })
        try:
            result, _, _ = module.run_local_esmfold2(
                sequence="ACDE",
                model_id_or_path="fixture-model",
                cache_dir=root,
                device="cpu",
                num_recycles=4,
                msa_a3m=msa_path,
                num_sampling_steps=17,
                num_diffusion_samples=2,
                seed=123,
            )
        finally:
            sys.modules.clear(); sys.modules.update(old)
        self.assertIsInstance(result, list)
        self.assertEqual(FakeBuilder.seen_kwargs["num_loops"], 4)
        self.assertEqual(FakeBuilder.seen_kwargs["num_sampling_steps"], 17)
        self.assertEqual(FakeBuilder.seen_kwargs["num_diffusion_samples"], 2)
        self.assertEqual(FakeBuilder.seen_kwargs["seed"], 123)

    def test_diffusion_samples_write_top1_and_sample_outputs(self):
        module = load_runner_module()
        root = TEST_TMP_ROOT / "esm_sampling_outputs"
        out_dir = root / "out"
        if out_dir.exists():
            for path in sorted(out_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

        class FakeResult:
            def __init__(self, label, ptm, plddt, pae):
                self.label = label
                self.ptm = ptm
                self.plddt = plddt
                self.pae = pae
            def to_pdb_string(self):
                return f"ATOM {self.label}\nEND\n"

        results = [
            FakeResult("low", 0.5, [0.7, 0.7], [5.0, 5.0]),
            FakeResult("high", 0.9, [0.8, 0.9], [2.0, 3.0]),
        ]
        best, best_index = module.select_best_result(results)
        self.assertIs(best, results[1])
        self.assertEqual(best_index, 1)
        for sample_index, sample in enumerate(results):
            module.write_prediction_outputs(
                name=f"design_sample{sample_index:02d}",
                sequence="ACDE",
                result=sample,
                pdb_text=None,
                msa_metadata={"msa_model_consumed": True},
                out_dir=out_dir / "samples",
                model_id_or_path="fixture",
                device="cpu",
                cache_dir=root,
                sample_index=sample_index,
                num_diffusion_samples=2,
                selected_sample_index=best_index,
                num_sampling_steps=17,
                seed=123,
            )
        module.write_prediction_outputs(
            name="design",
            sequence="ACDE",
            result=best,
            pdb_text=None,
            msa_metadata={"msa_model_consumed": True},
            out_dir=out_dir,
            model_id_or_path="fixture",
            device="cpu",
            cache_dir=root,
            sample_index=best_index,
            num_diffusion_samples=2,
            selected_sample_index=best_index,
            num_sampling_steps=17,
            seed=123,
        )
        self.assertTrue((out_dir / "design.pdb").is_file())
        self.assertTrue((out_dir / "samples" / "design_sample00.pdb").is_file())
        self.assertTrue((out_dir / "samples" / "design_sample01.pdb").is_file())
        self.assertIn("high", (out_dir / "design.pdb").read_text())


if __name__ == "__main__":
    unittest.main()
