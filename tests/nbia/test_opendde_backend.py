from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from jango.pipeline.fold_execution import prediction_outputs_complete
from nbia.folding import OpenDDEBackend, get_fold_backend
from nbia.folding.base import FoldJob
from nbia.folding.msa import MSAArtifact, MSAFormat, MSAPairing, MSAProviderKind, MSAProvenance, MSAStatus


class OpenDDEBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runner = self.root / "run_opendde_monomer.py"
        self.runner.write_text("#!/usr/bin/env python\n")
        self.runner.chmod(0o755)
        self.opendde_python = self.root / "opendde-python"
        self.opendde_python.write_text("stub")
        self.opendde_python.chmod(0o755)
        self.opendde = self.root / "opendde"
        self.opendde.write_text("#!/usr/bin/env bash\necho fixture\n")
        self.opendde.chmod(0o755)
        self.model_root = self.root / "models"
        self.model_root.mkdir()
        self.checkpoint = self.model_root / "opendde.pt"
        self.checkpoint.write_text("stub")
        self.msa = self.root / "query.a3m"
        self.msa.write_text(">query\nACDE\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def backend(self) -> OpenDDEBackend:
        return OpenDDEBackend(
            opendde_python=self.opendde_python,
            opendde_executable=self.opendde,
            root_dir=self.model_root,
            checkpoint=self.checkpoint,
            runner_path=self.runner,
        )

    def msa_artifact(self) -> MSAArtifact:
        return MSAArtifact(
            msa_id="msa1",
            sequence_hash="hash",
            query_sequence_hash="hash",
            structure_id="s1",
            backend="opendde",
            provider=MSAProviderKind.ABFORGE_GET_OR_BUILD,
            format=MSAFormat.A3M,
            path=self.msa,
            paired_or_unpaired=MSAPairing.UNPAIRED,
            chain_ids=("A",),
            sequence_count=1,
            generation_status=MSAStatus.CACHE_HIT,
            provenance=MSAProvenance(provider=MSAProviderKind.ABFORGE_GET_OR_BUILD),
            kind="unpaired",
            model_consumed=True,
        )

    def test_backend_registered(self) -> None:
        env = {
            "OPENDDE_PYTHON": str(self.opendde_python),
            "OPENDDE_EXECUTABLE": str(self.opendde),
            "OPENDDE_ROOT_DIR": str(self.model_root),
            "OPENDDE_RUNNER": str(self.runner),
        }
        with patch.dict(os.environ, env):
            self.assertIsInstance(get_fold_backend("open-dde"), OpenDDEBackend)

    def test_writes_json_with_unpaired_msa_and_command_requires_msa(self) -> None:
        backend = self.backend()
        input_path, output_dir = backend.write_input(
            design_id="s1_mpnn_0001",
            structure_id="s1",
            sequence="ACDE",
            work_dir=self.root / "work",
            msa_artifact=self.msa_artifact(),
        )
        payload = json.loads(input_path.read_text())
        chain = payload[0]["sequences"][0]["proteinChain"]
        self.assertEqual("ACDE", chain["sequence"])
        self.assertEqual(str(self.msa), chain["unpairedMsaPath"])
        command = backend.command(input_path, output_dir)
        self.assertIn("run_opendde_monomer.py", command)
        self.assertIn("--use-msa true", command)
        self.assertIn("--require-msa", command)
        self.assertIn("--checkpoint", command)

    def test_without_msa_is_explicit_sequence_only_command(self) -> None:
        backend = self.backend()
        input_path, output_dir = backend.write_input(
            design_id="s1_mpnn_0001",
            structure_id="s1",
            sequence="ACDE",
            work_dir=self.root / "work",
            msa_artifact=None,
        )
        command = backend.command(input_path, output_dir)
        self.assertIn("--use-msa false", command)
        self.assertNotIn("--require-msa", command)

    def test_job_bundle_uses_backend_tsv_name(self) -> None:
        backend = self.backend()
        input_path, output_dir = backend.write_input(
            design_id="s1_mpnn_0001",
            structure_id="s1",
            sequence="ACDE",
            work_dir=self.root / "work",
        )
        backend.write_job_bundle(
            [
                FoldJob(
                    design_id="s1_mpnn_0001",
                    structure_id="s1",
                    sequence="ACDE",
                    input_path=input_path,
                    output_dir=output_dir,
                    command=backend.command(input_path, output_dir),
                    backend="opendde",
                    metadata={"fold_backend": "opendde", "redesign_mode": "hotspot_only"},
                )
            ],
            self.root / "bundle",
        )
        tsv = self.root / "bundle" / "opendde_monomer_jobs.tsv"
        self.assertTrue(tsv.is_file())
        rows = pd.read_csv(tsv, sep="\t")
        self.assertEqual("opendde", rows.loc[0, "fold_backend"])
        self.assertTrue((self.root / "bundle" / "run_opendde_monomer_jobs.sh").is_file())

    def test_discovers_normalized_predictions_with_confidence(self) -> None:
        output_dir = self.root / "out"
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(parents=True)
        pdb = prediction_dir / "sample_0.pdb"
        pdb.write_text("ATOM\n")
        conf = prediction_dir / "sample_0_confidence.json"
        conf.write_text(json.dumps({"ranking_score": 0.8, "plddt": 91.2}))
        (output_dir / "opendde_prediction_manifest.csv").write_text(
            "prediction_path,source_cif,confidence_json,sample_rank,ranking_score\n"
            f"{pdb},raw.cif,{conf},0,0.8\n"
        )
        predictions = self.backend().discover_predictions(output_dir)
        self.assertEqual([pdb], [item.prediction_path for item in predictions])
        self.assertEqual(0.8, predictions[0].confidence["opendde_ranking_score"])
        self.assertEqual(91.2, predictions[0].confidence["opendde_plddt"])

    def test_raw_opendde_cif_alone_is_not_complete(self) -> None:
        output_dir = self.root / "out"
        raw = output_dir / "raw_opendde" / "job" / "seed_101" / "predictions"
        raw.mkdir(parents=True)
        (raw / "job_sample_0.cif").write_text("data_job\n")
        self.assertFalse(prediction_outputs_complete(output_dir))
        normalized = output_dir / "predictions"
        normalized.mkdir()
        (normalized / "job_sample_0.pdb").write_text("ATOM\n")
        self.assertTrue(prediction_outputs_complete(output_dir))


if __name__ == "__main__":
    unittest.main()
