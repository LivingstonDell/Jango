from __future__ import annotations

from pathlib import Path
import unittest

from jango.paths import RunPaths, recommended_run_root, resolve_output_root
from jango.pipeline.decoy_fold import fold_root
from jango.pipeline.decoy_prefold import prefold_paths
from jango.pipeline.nanobody_pipeline import native_paths
from jango.preprocessing.native_inputs import preprocessing_paths


class RunPathsTests(unittest.TestCase):
    def test_rejects_repository_internal_output_root(self) -> None:
        repo = Path("<JANGO_REPO>")
        with self.assertRaises(ValueError):
            resolve_output_root(repo / "results", repo_root=repo)

    def test_recommended_roots_use_work_directory(self) -> None:
        test_root = recommended_run_root("smoke", test=True, user="demo")
        production_root = recommended_run_root("smoke", test=False, user="demo")

        self.assertEqual(Path("/data/demo/jango/work/smoke"), test_root)
        self.assertEqual(test_root, production_root)

    def test_native_layout_uses_canonical_branches(self) -> None:
        root = resolve_output_root(Path("/data/demo/jango/work/unit"))
        paths = native_paths(root)
        pre = preprocessing_paths(root, "sabdab1_2")

        self.assertEqual(paths["manifest"], root / "data" / "manifests" / "native" / "manifest.csv")
        self.assertEqual(paths["features"], root / "results" / "native" / "tables" / "native_interface_features.csv")
        self.assertEqual(paths["work"], root / "work" / "native" / "rosetta")
        self.assertEqual(pre.tarballs, root / "data" / "raw" / "sabdab1_2" / "tarballs")

    def test_decoy_layout_exposes_canonical_artifacts(self) -> None:
        root = resolve_output_root(Path("/data/demo/jango/work/unit"))
        layout = RunPaths.from_output_root(root).decoy_layout("esmfold2_50_auto_by_length")

        self.assertEqual(layout.prefold_manifest, root / "data" / "manifests" / "decoy" / "esmfold2_50_auto_by_length" / "prefold_manifest.json")
        self.assertEqual(layout.source_case_manifest, root / "data" / "manifests" / "decoy" / "esmfold2_50_auto_by_length" / "source_case_manifest.csv")
        self.assertEqual(layout.fold_manifest, root / "data" / "manifests" / "decoy" / "esmfold2_50_auto_by_length" / "fold_manifest.json")
        self.assertEqual(layout.combined_fold_submission_manifest, root / "work" / "decoy" / "esmfold2_50_auto_by_length" / "slurm" / "fold_submission_manifest.csv")

    def test_decoy_layout_uses_singular_decoy(self) -> None:
        root = resolve_output_root(Path("/data/demo/jango/work/unit"))
        run = RunPaths.from_output_root(root)
        experiment = "boltz2_5_manual"
        mode = "hotspot_only"

        self.assertEqual(
            run.decoy_work_dir(experiment, mode),
            root / "work" / "decoy" / experiment / mode,
        )
        self.assertEqual(
            run.decoy_tables_dir(experiment, mode),
            root / "results" / "decoy" / "tables" / experiment / mode,
        )
        self.assertEqual(
            run.decoy_figures_dir(experiment, mode),
            root / "results" / "decoy" / "figures" / experiment / mode,
        )
        self.assertEqual(
            run.decoy_manifest_dir(experiment, mode),
            root / "data" / "manifests" / "decoy" / experiment / mode,
        )
        self.assertNotIn("decoys", str(run.decoy_work_dir(experiment, mode)))

    def test_stage_metadata_paths_are_manifest_based(self) -> None:
        root = resolve_output_root(Path("/data/demo/jango/work/unit"))
        experiment = "esmfold2_5_manual"
        prefold = prefold_paths(root, experiment)

        self.assertEqual(prefold["prefold_dir"], root / "data" / "manifests" / "decoy" / experiment)
        self.assertEqual(fold_root(root, experiment), prefold["prefold_dir"])


if __name__ == "__main__":
    unittest.main()