import unittest
from argparse import Namespace
from pathlib import Path

from jango.pipeline import decoy_analyze, nanobody_pipeline


class RosettaRuntimePassThroughTests(unittest.TestCase):
    def test_nanobody_pipeline_passes_apptainer_runtime_args(self):
        args = Namespace(
            rosetta_runtime="apptainer",
            rosetta_image="docker://rosettacommons/rosetta:latest",
            docker_image=None,
            rosetta_bin_dir=None,
            apptainer_cache_dir=Path("cache"),
            limit=2,
            jobs=4,
            contact_cutoff=5.0,
            backfill_fullscore=True,
        )
        paths = nanobody_pipeline.native_paths(Path("/data/demo/jango/work/unit"))
        commands = nanobody_pipeline.build_commands(args, paths, Path("raw"))
        relax_argv = commands[2].argv
        backfill_argv = commands[4].argv

        self.assertIn("--rosetta-runtime", relax_argv)
        self.assertIn("apptainer", relax_argv)
        self.assertIn("--rosetta-image", relax_argv)
        self.assertIn("docker://rosettacommons/rosetta:latest", relax_argv)
        self.assertIn("--apptainer-cache-dir", relax_argv)
        self.assertIn(Path("cache"), relax_argv)
        self.assertIn("--rosetta-runtime", backfill_argv)

    def test_decoy_analyze_passes_local_runtime_args(self):
        rosetta_args = ["--rosetta-runtime", "local", "--rosetta-bin-dir", Path("/opt/rosetta/bin")]
        commands = decoy_analyze.build_mode_commands(
            mode="full_antigen",
            case_manifest=Path("case.csv"),
            validation=Path("validation.csv"),
            raw_dir=Path("raw"),
            tables_dir=Path("tables"),
            figures_dir=Path("figures"),
            work_dir=Path("work"),
            native_rosetta=Path("native_rosetta.csv"),
            native_features=Path("native_features.csv"),
            native_relax_manifest=None,
            dockq_bin=None,
            allow_dockq_failures=False,
            dockq_skip_existing=False,
            skip_dockq=True,
            experiment="test_experiment",
            rosetta_args=rosetta_args,
            jobs=2,
            limit=3,
        )
        relax_argv = commands[1].argv
        metrics_argv = commands[2].argv

        self.assertIn("--rosetta-runtime", relax_argv)
        self.assertIn("local", relax_argv)
        self.assertIn("--rosetta-bin-dir", metrics_argv)
        self.assertIn(Path("/opt/rosetta/bin"), metrics_argv)
        self.assertNotIn("--docker-image", relax_argv)


if __name__ == "__main__":
    unittest.main()

