import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from nbia.config import RosettaRuntimeConfig
from nbia.folding import (
    Boltz2Backend,
    ESMFold2Backend,
    MSACache,
    MSACapability,
    MSAConfig,
    MSAError,
    MSAProviderKind,
    MSARequest,
    MSAStatus,
    MSAUnsupportedError,
    MsaInput,
    MsaPolicy,
)
from nbia.folding.msa import discover_boltz_backend_managed_msa, resolve_msa_for_backend
from nbia.runtime import RosettaCommand


TEST_TMP_ROOT = Path(os.environ.get("JANGO_TEST_TMP", Path(tempfile.gettempdir()) / "jango_test_tmp" / "phase2_interfaces"))


def workspace_case_root(name: str) -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Phase2InterfaceTests(unittest.TestCase):
    def test_msa_policy_rejects_missing_required_msa(self):
        policy = MsaPolicy(requirement="required")
        with self.assertRaises(ValueError):
            policy.validate_for_backend("esmfold2", None)

    def test_msa_policy_accepts_optional_msa(self):
        policy = MsaPolicy(requirement="optional")
        msa = MsaInput(path=Path("example.a3m"), format="a3m", source="fixture")
        policy.validate_for_backend("boltz2", msa)

    def test_backend_msa_capability_declarations(self):
        self.assertEqual(Boltz2Backend.msa_support.capability, MSACapability.BACKEND_MANAGED)
        self.assertTrue(Boltz2Backend.msa_support.generates_msa)
        self.assertFalse(Boltz2Backend.msa_support.accepts_precomputed)
        self.assertEqual(ESMFold2Backend.msa_support.capability, MSACapability.OPTIONAL)
        self.assertTrue(ESMFold2Backend.msa_support.accepts_precomputed)

    def test_boltz_command_toggles_backend_managed_msa_server(self):
        root = workspace_case_root("boltz_msa_command")
        backend = Boltz2Backend(use_msa_server=True, boltz_executable="boltz")
        with_msa = backend.command(root / "in.yaml", root / "out")
        self.assertIn("--use_msa_server", with_msa)
        backend_no_msa = Boltz2Backend(use_msa_server=False, boltz_executable="boltz")
        without_msa = backend_no_msa.command(root / "in.yaml", root / "out")
        self.assertNotIn("--use_msa_server", without_msa)

    def test_cache_key_is_deterministic_for_same_sequence_request(self):
        request_a = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACD E",
            backend="boltz2",
            provider=MSAProviderKind.PRECOMPUTED,
        )
        request_b = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACDE",
            backend="boltz2",
            provider=MSAProviderKind.PRECOMPUTED,
        )
        self.assertEqual(request_a.query_sequence_hash, request_b.query_sequence_hash)
        self.assertEqual(request_a.request_hash, request_b.request_hash)

    def test_precomputed_msa_loads_and_writes_manifest(self):
        root = workspace_case_root("precomputed_msa")
        msa_path = root / "inputs" / "example.a3m"
        msa_path.parent.mkdir(parents=True, exist_ok=True)
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")
        request = MSARequest(
            structure_id="s1",
            design_id="s1_mpnn_0001",
            sequence="ACDE",
            backend="fixture",
            provider=MSAProviderKind.PRECOMPUTED,
        )
        artifact = MSACache(root / "msa").load_precomputed(request, msa_path)
        self.assertEqual(artifact.sequence_count, 2)
        self.assertEqual(artifact.generation_status, MSAStatus.CACHE_HIT)
        self.assertTrue(artifact.manifest_path and artifact.manifest_path.exists())
        manifest = artifact.to_manifest_dict(base_dir=root)
        self.assertTrue(manifest["relative_path"].startswith("inputs"))

    def test_invalid_precomputed_msa_rejects_missing_query_sequence(self):
        root = workspace_case_root("invalid_precomputed_msa")
        msa_path = root / "bad.a3m"
        msa_path.write_text(">hit\nYYYY\n")
        request = MSARequest(
            structure_id="s1",
            design_id="s1_mpnn_0001",
            sequence="ACDE",
            backend="fixture",
            provider=MSAProviderKind.PRECOMPUTED,
        )
        with self.assertRaises(MSAError):
            MSACache(root / "msa").load_precomputed(request, msa_path)

    def test_abforge_get_or_build_provider_uses_lookup_json_cached_a3m(self):
        root = workspace_case_root("abforge_lookup")
        msa_path = root / "cache" / "by_sequence_hash" / "unused" / "unpaired.a3m"
        msa_path.parent.mkdir(parents=True, exist_ok=True)
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")
        helper = root / "get_or_build_msa.py"
        helper.write_text(
            "import json\n"
            f"print(json.dumps({{'queries': [{{'status': 'complete', 'unpaired_msa_path': {str(msa_path)!r}}}]}}))\n"
        )
        config = MSAConfig.from_values(
            mode="required",
            provider="abforge_get_or_build",
            cache_dir=root / "cache",
            format="a3m",
            pairing="unpaired",
            script_path=helper,
            tool="esmfold2",
            build_if_missing=False,
        )
        request = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACDE",
            backend="esmfold2",
            provider=MSAProviderKind.ABFORGE_GET_OR_BUILD,
        )
        resolution = resolve_msa_for_backend(
            config=config,
            support=ESMFold2Backend.msa_support,
            request=request,
            backend_name="esmfold2",
        )
        self.assertEqual(resolution.status, MSAStatus.CACHE_HIT)
        assert resolution.artifact is not None
        self.assertEqual(resolution.artifact.provider, MSAProviderKind.ABFORGE_GET_OR_BUILD)
        self.assertEqual(resolution.artifact.path, msa_path.resolve())
        self.assertTrue(resolution.artifact.model_consumed)
        self.assertEqual(resolution.artifact.provenance.tool_name, "abforge_get_or_build_msa.py")

    def test_esmfold2_required_precomputed_msa_resolves_to_consumed_artifact(self):
        root = workspace_case_root("esm_supported")
        msa_path = root / "example.a3m"
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")
        config = MSAConfig.from_values(
            mode="required",
            provider="precomputed",
            cache_dir=root / "msa",
            input_path=msa_path,
            format="a3m",
        )
        request = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACDE",
            backend="esmfold2",
            provider=MSAProviderKind.PRECOMPUTED,
        )
        resolution = resolve_msa_for_backend(
            config=config,
            support=ESMFold2Backend.msa_support,
            request=request,
            backend_name="esmfold2",
        )
        self.assertEqual(resolution.status, MSAStatus.CACHE_HIT)
        self.assertIsNotNone(resolution.artifact)
        assert resolution.artifact is not None
        self.assertTrue(resolution.artifact.model_consumed)
        fields = resolution.to_manifest_fields(base_dir=root)
        self.assertTrue(fields["msa_model_consumed"])
        self.assertEqual(fields["msa_kind"], "unpaired")

    def test_esmfold2_backend_writes_msa_sidecar_and_command(self):
        root = workspace_case_root("esm_backend_msa_sidecar")
        msa_path = root / "example.a3m"
        msa_path.write_text(">query\nACDE\n>hit\nAC-E\n")
        config = MSAConfig.from_values(
            mode="required",
            provider="precomputed",
            cache_dir=root / "msa",
            input_path=msa_path,
            format="a3m",
        )
        request = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACDE",
            backend="esmfold2",
            provider=MSAProviderKind.PRECOMPUTED,
        )
        resolution = resolve_msa_for_backend(
            config=config,
            support=ESMFold2Backend.msa_support,
            request=request,
            backend_name="esmfold2",
        )
        assert resolution.artifact is not None
        backend = ESMFold2Backend(
            esmfold2_python="/usr/bin/python",
            esm_root=root / "esm",
            model_id_or_path="fixture",
            cache_dir=root / "cache",
            device="cpu",
        )
        fasta, out_dir = backend.write_input(
            design_id="s1_native",
            structure_id="s1",
            sequence="ACDE",
            work_dir=root / "work",
            msa_artifact=resolution.artifact,
        )
        sidecar = backend.msa_sidecar_path(fasta)
        self.assertTrue(sidecar.exists())
        command = backend.command(fasta, out_dir)
        self.assertIn("--msa-a3m", command)
        self.assertIn(str(msa_path.resolve()), command)
        self.assertIn("--require-msa", command)

    def test_optional_unsupported_msa_can_use_explicit_fallback(self):
        config = MSAConfig.from_values(
            mode="optional",
            provider="none",
            allow_single_sequence_fallback=True,
        )
        request = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACDE",
            backend="esmfold2",
            provider=MSAProviderKind.NONE,
        )
        resolution = resolve_msa_for_backend(
            config=config,
            support=ESMFold2Backend.msa_support,
            request=request,
            backend_name="esmfold2",
        )
        self.assertEqual(resolution.status, MSAStatus.SINGLE_SEQUENCE_EXPLICIT_FALLBACK)

    def test_boltz_backend_managed_resolution_and_artifact_discovery(self):
        root = workspace_case_root("boltz_backend_managed")
        config = MSAConfig.from_values(mode="backend-managed", provider="backend-managed")
        request = MSARequest(
            structure_id="s1",
            design_id="s1_native",
            sequence="ACDE",
            backend="boltz2",
            provider=MSAProviderKind.BACKEND_MANAGED,
        )
        resolution = resolve_msa_for_backend(
            config=config,
            support=Boltz2Backend.msa_support,
            request=request,
            backend_name="boltz2",
        )
        self.assertEqual(resolution.status, MSAStatus.BACKEND_MANAGED)

        msa_dir = root / "out" / "boltz_results_s1_native" / "msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        (msa_dir / "s1_native_0.csv").write_text("key,sequence\n-1,ACDE\n-1,AC-E\n")
        discovered = discover_boltz_backend_managed_msa(output_dir=root / "out", request=request, config=config)
        self.assertIsNotNone(discovered)
        assert discovered is not None
        self.assertEqual(discovered.status, MSAStatus.BACKEND_MANAGED)
        self.assertEqual(discovered.artifact.sequence_count, 2)

    def test_rosetta_runtime_config_validation(self):
        cfg = RosettaRuntimeConfig(kind="docker", image="rosettacommons/rosetta:latest")
        cfg.validate()
        command = RosettaCommand(argv=("rosetta_scripts", "-help"), work_dir=Path("work"), runtime=cfg)
        self.assertEqual(command.runtime.kind, "docker")


if __name__ == "__main__":
    unittest.main()
