import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jango.runtime.rosetta import (
    RosettaPreflightError,
    RosettaRuntimeConfig,
    validate_rosetta_runtime,
)


class RosettaPreflightTests(unittest.TestCase):
    def test_apptainer_docker_uri_uses_exec_not_inspect(self):
        config = RosettaRuntimeConfig(
            kind="apptainer",
            image="docker://rosettacommons/rosetta:latest",
            apptainer_cache_dir=Path("/cache/apptainer"),
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("jango.runtime.rosetta.subprocess.run", return_value=completed) as run:
            result = validate_rosetta_runtime(config, apptainer_bin="apptainer", timeout_seconds=1)

        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["apptainer", "exec", "docker://rosettacommons/rosetta:latest"])
        self.assertNotIn("inspect", argv)
        self.assertEqual(argv[3:5], ["sh", "-lc"])
        self.assertIn("test -x /usr/local/bin/InterfaceAnalyzer.default.linuxgccrelease", argv[-1])
        self.assertIn("test -x /usr/local/bin/rosetta_scripts.default.linuxgccrelease", argv[-1])
        self.assertEqual(run.call_args.kwargs["env"]["APPTAINER_CACHEDIR"], "/cache/apptainer")
        self.assertEqual(result.method, "apptainer exec docker URI executable check")

    def test_apptainer_local_sif_uses_exec(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "rosetta.sif"
            image.write_text("stub")
            config = RosettaRuntimeConfig(kind="apptainer", image=str(image))
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            with patch("jango.runtime.rosetta.subprocess.run", return_value=completed) as run:
                result = validate_rosetta_runtime(config, apptainer_bin="apptainer", timeout_seconds=1)

        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["apptainer", "exec", str(image)])
        self.assertNotIn("inspect", argv)
        self.assertEqual(result.method, "apptainer exec local image executable check")

    def test_apptainer_local_sif_must_exist(self):
        config = RosettaRuntimeConfig(kind="apptainer", image="/missing/rosetta.sif")
        with self.assertRaises(RosettaPreflightError):
            validate_rosetta_runtime(config, apptainer_bin="apptainer", timeout_seconds=1)

    def test_local_runtime_checks_required_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            (bin_dir / "InterfaceAnalyzer.default.linuxgccrelease").write_text("stub")
            (bin_dir / "rosetta_scripts.default.linuxgccrelease").write_text("stub")
            config = RosettaRuntimeConfig(kind="local", bin_dir=bin_dir)
            result = validate_rosetta_runtime(config)

        self.assertEqual(result.method, "local executable check")


if __name__ == "__main__":
    unittest.main()
