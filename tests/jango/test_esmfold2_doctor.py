import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jango import cli
from jango.paths import repository_root


EXPECTED_CONFIGS = [
    "environment.yml",
    "examples/paths.example.env",
    "examples/runtime.<SLURM_NODE>.example.env",
    "test/paths.env",
    "test/esmfold2.env",
    "production/paths.env",
    "production/runtime.<SLURM_NODE>.common.env",
    "production/esmfold2.env",
    "production/boltz2.env",
    "production/opendde.env",
]


def write_expected_configs(root: Path) -> None:
    for rel in EXPECTED_CONFIGS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("export OK=1\n")


class JangoDoctorTests(unittest.TestCase):
    def test_doctor_reports_grouped_esmfold2_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_root = root / "configs"
            write_expected_configs(config_root)
            paths_config = config_root / "test" / "paths.env"
            runtime_config = config_root / "test" / "esmfold2.env"
            output_root = root / "run"
            tools = root / "tools"
            proteinmpnn_home = tools / "ProteinMPNN"
            proteinmpnn_home.mkdir(parents=True)
            proteinmpnn_python = tools / "proteinmpnn-python"
            proteinmpnn_python.write_text("stub")
            proteinmpnn_runner = proteinmpnn_home / "protein_mpnn_run.py"
            proteinmpnn_runner.write_text("stub")
            esm_python = tools / "esm-python"
            esm_python.write_text("stub")
            esm_root = tools / "esm-root"
            esm_root.mkdir()
            (repository_root() / "scripts" / "nbia" / "run_esmfold2_monomer.py").touch(exist_ok=True)
            msa_script = tools / "get_or_build_msa.py"
            msa_script.write_text("stub")
            msa_cache = tools / "msa_cache"
            msa_cache.mkdir()
            env = {
                "CONDA_DEFAULT_ENV": "jango",
                "JANGO_SOURCE": str(repository_root()),
                "JANGO_TEST_RUN_ROOT": str(output_root),
                "JANGO_PATHS_CONFIG": str(paths_config),
                "JANGO_RUNTIME_CONFIG": str(runtime_config),
                "FOLDING_BACKEND": "esmfold2",
                "PROTEINMPNN_HOME": str(proteinmpnn_home),
                "PROTEINMPNN_PYTHON": str(proteinmpnn_python),
                "PROTEINMPNN_RUNNER": str(proteinmpnn_runner),
                "ESMFOLD2_PYTHON": str(esm_python),
                "ESM_ROOT": str(esm_root),
                "MSA_REQUIRE": "true",
                "MSA_PROVIDER": "abforge_get_or_build",
                "MSA_SCRIPT": str(msa_script),
                "MSA_CACHE_ROOT": str(msa_cache),
            }
            with patch.dict(os.environ, env, clear=True), patch("builtins.print") as printer:
                rc = cli._doctor(["--quick", "--config-root", str(config_root)])
        self.assertEqual(rc, 0)
        printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
        self.assertIn("PASS", printed)
        self.assertIn("WARN", printed)
        self.assertIn("FAIL", printed)
        self.assertIn("ESMFold2 MSA API", printed)
        self.assertIn("repository-root generated leakage", printed)

    def test_doctor_fails_required_output_root_inside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_root = Path(tmp) / "configs"
            write_expected_configs(config_root)
            env = {
                "CONDA_DEFAULT_ENV": "jango",
                "JANGO_SOURCE": str(repository_root()),
                "JANGO_TEST_RUN_ROOT": str(repository_root() / "results"),
            }
            with patch.dict(os.environ, env, clear=True), patch("builtins.print"):
                rc = cli._doctor(["--quick", "--config-root", str(config_root)])
        self.assertEqual(rc, 1)

    def test_doctor_skips_inactive_backend_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_root = root / "configs"
            write_expected_configs(config_root)
            tools = root / "tools"
            proteinmpnn_home = tools / "ProteinMPNN"
            proteinmpnn_home.mkdir(parents=True)
            proteinmpnn_python = tools / "proteinmpnn-python"
            proteinmpnn_python.write_text("stub")
            proteinmpnn_runner = proteinmpnn_home / "protein_mpnn_run.py"
            proteinmpnn_runner.write_text("stub")
            esm_python = tools / "esm-python"
            esm_python.write_text("stub")
            esm_root = tools / "esm-root"
            esm_root.mkdir()
            msa_script = tools / "get_or_build_msa.py"
            msa_script.write_text("stub")
            msa_cache = tools / "msa_cache"
            msa_cache.mkdir()
            env = {
                "CONDA_DEFAULT_ENV": "jango",
                "JANGO_SOURCE": str(repository_root()),
                "JANGO_TEST_RUN_ROOT": str(root / "run"),
                "FOLDING_BACKEND": "esmfold2",
                "PROTEINMPNN_HOME": str(proteinmpnn_home),
                "PROTEINMPNN_PYTHON": str(proteinmpnn_python),
                "PROTEINMPNN_RUNNER": str(proteinmpnn_runner),
                "ESMFOLD2_PYTHON": str(esm_python),
                "ESM_ROOT": str(esm_root),
                "MSA_REQUIRE": "true",
                "MSA_PROVIDER": "abforge_get_or_build",
                "MSA_SCRIPT": str(msa_script),
                "MSA_CACHE_ROOT": str(msa_cache),
            }
            with patch.dict(os.environ, env, clear=True), patch("builtins.print") as printer:
                rc = cli._doctor(["--quick", "--config-root", str(config_root)])
        self.assertEqual(rc, 0)
        printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
        self.assertIn("ESMFOLD2_PYTHON", printed)
        self.assertNotIn("BOLTZ_EXECUTABLE", printed)
        self.assertNotIn("OPENDDE_EXECUTABLE", printed)


if __name__ == "__main__":
    unittest.main()
