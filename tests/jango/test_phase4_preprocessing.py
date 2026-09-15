import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from jango.pipeline import nanobody_pipeline
from jango.pipeline.labels import LabelConfig
from jango.preprocessing.native_inputs import (
    canonicalize_preprocessing_manifest,
    prepare_native_inputs,
    preprocessing_paths,
    validate_preprocessing_options,
)


class Phase4PreprocessingTests(unittest.TestCase):
    def test_raw_preprocessing_requires_metadata(self):
        with self.assertRaises(ValueError):
            validate_preprocessing_options(
                input_kind="sabdab2-cif",
                input_path=Path("cif"),
                metadata=None,
            )
        validate_preprocessing_options(
            input_kind="tarballs",
            input_path=Path("tarballs"),
            metadata=None,
        )

    def test_tarball_mode_rejects_metadata(self):
        with self.assertRaises(ValueError):
            validate_preprocessing_options(
                input_kind="tarballs",
                input_path=Path("tarballs"),
                metadata=Path("metadata.csv"),
            )

    def test_canonical_preprocessing_manifest_replaces_legacy_labels(self):
        raw = pd.DataFrame(
            {
                "structure_id": ["s1"],
                "label_type": ["old_role"],
                "label_confidence": ["old_confidence"],
                "source": ["old_source"],
                "label": [1],
            }
        )
        labels = LabelConfig(
            source="sabdab1_2",
            label="positive",
            role="native_complex",
            confidence="user_supplied",
        )
        out = canonicalize_preprocessing_manifest(raw, labels)
        self.assertEqual(out.loc[0, "source"], "sabdab1_2")
        self.assertEqual(out.loc[0, "label"], "positive")
        self.assertEqual(out.loc[0, "role"], "native_complex")
        self.assertEqual(out.loc[0, "confidence"], "user_supplied")
        self.assertNotIn("label_type", out.columns)
        self.assertNotIn("label_confidence", out.columns)

    def test_tarball_limit_stages_input_subset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for name in ["1ccc_H_A.pdb.tar.gz", "1aaa_H_A.pdb.tar.gz", "1bbb_H_A.pdb.tar.gz"]:
                (raw / name).write_bytes(b"fixture")
            labels = LabelConfig(
                source="sabdab1_2",
                label="positive",
                role="native_complex",
                confidence="user_supplied",
            )

            prepared = prepare_native_inputs(
                input_kind="tarballs",
                input_path=raw,
                output_root=root / "run",
                labels=labels,
                limit=2,
            )

            expected = preprocessing_paths(root / "run", "sabdab1_2").tarballs
            self.assertEqual(expected, prepared.raw_dir)
            self.assertEqual(["1aaa_H_A.pdb.tar.gz", "1bbb_H_A.pdb.tar.gz"], sorted(path.name for path in expected.glob("*.pdb.tar.gz")))
            self.assertFalse((expected / "1ccc_H_A.pdb.tar.gz").exists())

    def test_nanobody_pipeline_plans_raw_preprocessing_tarball_dir(self):
        parser = nanobody_pipeline.build_parser()
        args = parser.parse_args(
            [
                "--input", "raw_cif",
                "--input-kind", "sabdab2-cif",
                "--metadata", "metadata.csv",
                "--output-root", "/data/demo/jango/work/unit",
                "--atlas-root", "atlas",
                "--source", "sabdab1_2",
                "--label", "positive",
                "--role", "native_complex",
                "--confidence", "user_supplied",
                "--rosetta-runtime", "docker",
                "--plan-only",
            ]
        )
        output_root = Path("/data/demo/jango/work/unit")
        paths = preprocessing_paths(output_root, "sabdab1_2")
        self.assertEqual(nanobody_pipeline.planned_raw_dir(args, output_root, "sabdab1_2"), paths.tarballs)


if __name__ == "__main__":
    unittest.main()
