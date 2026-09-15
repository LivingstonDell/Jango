import argparse
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

TEST_ROOT = Path(os.environ.get("JANGO_TEST_TMP", Path(tempfile.gettempdir()) / "jango_test_tmp")) / "nbia_rosetta_runtime_manual"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEST_ROOT / ".matplotlib"))

from nbia.cli import rosetta_runtime_config_from_args
from nbia.config import RosettaRuntimeConfig
from nbia.rosetta import (
    _interface_analyzer_one,
    build_fullscore_command,
    build_interface_analyzer_command,
    build_relax_command,
    expected_relax_output_pdb,
)
from nbia.runtime.rosetta import (
    ApptainerRosettaRuntime,
    DockerRosettaRuntime,
    LocalRosettaRuntime,
    RosettaCommand,
    RosettaRuntimeError,
)


def make_root(name: str) -> Path:
    root = TEST_ROOT / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class RosettaRuntimeTests(unittest.TestCase):
    def test_docker_interface_analyzer_command(self):
        root = make_root("docker_interface")
        pdb = root / "input" / "model.pdb"
        score_dir = root / "scores"
        pdb.parent.mkdir()
        score_dir.mkdir()
        runtime = DockerRosettaRuntime(RosettaRuntimeConfig(kind="docker", image="rosetta:test"))

        command = build_interface_analyzer_command(
            runtime,
            pdb=pdb,
            score_dir=score_dir,
            scorefile=score_dir / "model.sc",
            interface="H_A",
        )

        self.assertEqual(command.argv[:3], ("docker", "run", "--rm"))
        self.assertIn(f"{pdb.parent.resolve()}:/input:ro", command.argv)
        self.assertIn(f"{score_dir.resolve()}:/scores", command.argv)
        self.assertIn("rosetta:test", command.argv)
        self.assertIn("/usr/local/bin/InterfaceAnalyzer.default.linuxgccrelease", command.argv)
        self.assertIn("/input/model.pdb", command.argv)
        self.assertIn("/scores/model.sc", command.argv)
        self.assertIn("-out:path:pdb", command.argv)
        self.assertIn("/scores/pdbs", command.argv)
        self.assertNotIn("-out:nooutput", command.argv)
        self.assertIn("-w", command.argv)
        self.assertEqual(command.argv[command.argv.index("-w") + 1], "/scores")
        self.assertEqual(command.work_dir, score_dir)

    def test_apptainer_interface_analyzer_command(self):
        root = make_root("apptainer_interface")
        pdb = root / "input" / "model.pdb"
        score_dir = root / "scores"
        pdb.parent.mkdir()
        score_dir.mkdir()
        runtime = ApptainerRosettaRuntime(
            RosettaRuntimeConfig(kind="apptainer", image="docker://rosettacommons/rosetta:latest", apptainer_cache_dir=root / "cache")
        )

        command = build_interface_analyzer_command(
            runtime,
            pdb=pdb,
            score_dir=score_dir,
            scorefile=score_dir / "model.sc",
            interface="H_A",
        )

        self.assertEqual(command.argv[:2], ("apptainer", "exec"))
        self.assertIn("--bind", command.argv)
        self.assertIn(f"{pdb.parent.resolve()}:/input:ro", command.argv)
        self.assertIn("docker://rosettacommons/rosetta:latest", command.argv)
        self.assertIn("/usr/local/bin/InterfaceAnalyzer.default.linuxgccrelease", command.argv)
        self.assertIn("-out:path:pdb", command.argv)
        self.assertIn("/scores/pdbs", command.argv)
        self.assertNotIn("-out:nooutput", command.argv)
        self.assertIn("--pwd", command.argv)
        self.assertEqual(command.argv[command.argv.index("--pwd") + 1], "/scores")
        self.assertEqual(command.work_dir, score_dir)
        self.assertEqual(command.env["APPTAINER_CACHEDIR"], str(root / "cache"))

    def test_local_executable_discovery_and_command(self):
        root = make_root("local_discovery")
        bin_dir = root / "bin"
        input_dir = root / "input"
        score_dir = root / "scores"
        bin_dir.mkdir()
        input_dir.mkdir()
        score_dir.mkdir()
        exe = bin_dir / "InterfaceAnalyzer.default.linuxgccrelease"
        exe.write_text("stub\n")
        pdb = input_dir / "model.pdb"
        runtime = LocalRosettaRuntime(RosettaRuntimeConfig(kind="local", bin_dir=bin_dir))

        command = build_interface_analyzer_command(
            runtime,
            pdb=pdb,
            score_dir=score_dir,
            scorefile=score_dir / "model.sc",
            interface="H_A",
        )

        self.assertEqual(Path(command.argv[0]), exe)
        self.assertIn(str(pdb), command.argv)
        self.assertIn(str(score_dir / "model.sc"), command.argv)
        self.assertIn("-out:path:pdb", command.argv)
        self.assertIn(str(score_dir / "pdbs"), command.argv)
        self.assertNotIn("-out:nooutput", command.argv)
        self.assertEqual(command.work_dir, score_dir)

    def test_missing_container_runtime_errors(self):
        runtime = DockerRosettaRuntime(RosettaRuntimeConfig(kind="docker", image="rosetta:test"))
        with patch("nbia.runtime.rosetta.shutil.which", return_value=None):
            with self.assertRaises(RosettaRuntimeError):
                runtime.check_available()

    def test_relax_and_decoy_relax_command_generation(self):
        work = make_root("relax") / "work"
        xml = work / "rosetta_xml" / "s1.xml"
        pdb = work / "extracted" / "s1.pdb"
        out = work / "relaxed"
        for path in [xml.parent, pdb.parent, out]:
            path.mkdir(parents=True, exist_ok=True)
        runtime = ApptainerRosettaRuntime(RosettaRuntimeConfig(kind="apptainer", image="docker://rosetta"))

        native = build_relax_command(runtime, work_dir=work, xml=xml, pdb=pdb, output_dir=out)
        decoy = build_relax_command(runtime, work_dir=work, xml=xml, pdb=pdb, output_dir=out, constrain_to_start=True)

        self.assertIn("/usr/local/bin/rosetta_scripts.default.linuxgccrelease", native.argv)
        self.assertIn("/work/rosetta_xml/s1.xml", native.argv)
        self.assertIn("/work/extracted/s1.pdb", native.argv)
        self.assertIn("/work/relaxed", native.argv)
        self.assertIn("--pwd", native.argv)
        self.assertEqual(native.argv[native.argv.index("--pwd") + 1], "/work")
        self.assertEqual(native.work_dir, work)
        self.assertIn("-relax:constrain_relax_to_start_coords", decoy.argv)

    def test_expected_relax_output_pdb_avoids_doubled_suffix(self):
        out = Path("relaxed")
        self.assertEqual(expected_relax_output_pdb(out, "s1"), out / "s1_interface_relax_0001.pdb")
        self.assertEqual(expected_relax_output_pdb(out, "s1_interface_relax_0001"), out / "s1_interface_relax_0001.pdb")

    def test_fullscore_command_generation(self):
        root = make_root("fullscore")
        pdb = root / "input" / "model.pdb"
        score_dir = root / "scores"
        pdb.parent.mkdir()
        score_dir.mkdir()
        runtime = DockerRosettaRuntime(RosettaRuntimeConfig(kind="docker", image="rosetta:test"))
        command = build_fullscore_command(runtime, pdb=pdb, score_dir=score_dir, scorefile=score_dir / "model.sc")
        self.assertIn("/usr/local/bin/score_jd2.default.linuxgccrelease", command.argv)
        self.assertIn("/input/model.pdb", command.argv)
        self.assertIn("/scores/model.sc", command.argv)
        self.assertIn("-out:nooutput", command.argv)
        self.assertIn("-w", command.argv)
        self.assertEqual(command.argv[command.argv.index("-w") + 1], "/scores")
        self.assertEqual(command.work_dir, score_dir)

    def test_runtime_run_uses_command_work_dir(self):
        root = make_root("run_cwd")
        bin_dir = root / "bin"
        bin_dir.mkdir()
        runtime = LocalRosettaRuntime(RosettaRuntimeConfig(kind="local", bin_dir=bin_dir))
        command = RosettaCommand(argv=("echo", "ok"), work_dir=root, runtime=runtime.config)
        with patch("nbia.runtime.rosetta.subprocess.run", return_value=subprocess.CompletedProcess(command.argv, 0)) as run:
            runtime.run(command)
        self.assertEqual(run.call_args.kwargs["cwd"], str(root.resolve()))

    def test_repository_root_work_dir_is_rejected(self):
        repo_root = Path(__file__).resolve().parents[2]
        runtime = DockerRosettaRuntime(RosettaRuntimeConfig(kind="docker", image="rosetta:test"))
        with self.assertRaisesRegex(RosettaRuntimeError, "source repository root"):
            runtime.build_command("InterfaceAnalyzer", [], work_dir=repo_root)

    def test_forced_chain_retry_behavior(self):
        root = make_root("forced_retry")
        work = root / "work"
        score_dir = root / "scores"
        relaxed = work / "relaxed" / "s1_interface_relax_0001.pdb"
        scorefile = score_dir / "s1.sc"
        relaxed.parent.mkdir(parents=True)
        score_dir.mkdir()
        relaxed.write_text("ATOM\n")
        runtime = DockerRosettaRuntime(RosettaRuntimeConfig(kind="docker", image="rosetta:test"))
        calls = []

        def fake_run(command, stdout=None, stderr=None):
            calls.append(command.argv)
            if len(calls) == 1:
                stdout.write("pose has only one chain\n")
                return subprocess.CompletedProcess(command.argv, 1)
            scorefile.write_text(
                "SCORE: dG_separated dSASA_int sc_value hbonds_int delta_unsatHbonds packstat description\n"
                "SCORE: -1.0 100.0 0.5 2.0 1.0 0.1 s1\n"
            )
            return subprocess.CompletedProcess(command.argv, 0)

        runtime.run = fake_run  # type: ignore[method-assign]
        row = {"structure_id": "s1", "pdb_id": "pdb1", "nanobody_chain": "H", "antigen_chain": "A"}
        result = _interface_analyzer_one(row, root / "raw", work, score_dir, "relaxed", runtime)

        self.assertEqual(result["interface_analyzer_status"], "ok_forced_chain_separation")
        self.assertEqual(len(calls), 2)
        self.assertIn("-in:file:treat_residues_in_these_chains_as_separate_chemical_entities", calls[1])
        self.assertIn("HA", calls[1])
        self.assertIn("-out:path:pdb", calls[0])
        self.assertIn("-out:path:pdb", calls[1])
        self.assertNotIn("-out:nooutput", calls[0])
        self.assertNotIn("-out:nooutput", calls[1])

    def test_cli_runtime_config_and_docker_alias(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            rosetta_runtime="apptainer",
            rosetta_image="docker://rosettacommons/rosetta:latest",
            docker_image=None,
            rosetta_bin_dir=None,
            apptainer_cache_dir=Path("cache"),
        )
        cfg = rosetta_runtime_config_from_args(parser, args)
        self.assertEqual(cfg.kind, "apptainer")
        self.assertEqual(cfg.apptainer_cache_dir, Path("cache"))

        legacy = argparse.Namespace(
            rosetta_runtime=None,
            rosetta_image=None,
            docker_image="rosetta:legacy",
            rosetta_bin_dir=None,
            apptainer_cache_dir=None,
        )
        cfg = rosetta_runtime_config_from_args(parser, legacy)
        self.assertEqual(cfg.kind, "docker")
        self.assertEqual(cfg.image, "rosetta:legacy")


if __name__ == "__main__":
    unittest.main()


