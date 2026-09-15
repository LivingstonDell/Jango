from __future__ import annotations

import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

from nbia import cli


class NbiaOutputGuardTests(unittest.TestCase):
    def test_blocks_generated_output_inside_source_repo(self) -> None:
        repo = Path("<JANGO_REPO>").resolve()
        args = argparse.Namespace(
            command="manifest",
            out=repo / "data" / "manifest" / "manifest.csv",
            checksums=repo / "data" / "manifest" / "checksums.sha256",
        )

        with patch.object(cli, "SOURCE_REPO_ROOT", repo):
            with self.assertRaises(ValueError):
                cli.guard_generated_outputs_outside_source_repo(args)

    def test_allows_generated_output_outside_source_repo(self) -> None:
        repo = Path("<JANGO_REPO>").resolve()
        run = Path("/data/demo/jango/work/smoke").resolve()
        args = argparse.Namespace(
            command="manifest",
            out=run / "data" / "manifests" / "native" / "manifest.csv",
            checksums=run / "data" / "manifests" / "native" / "checksums.sha256",
        )

        with patch.object(cli, "SOURCE_REPO_ROOT", repo):
            cli.guard_generated_outputs_outside_source_repo(args)


if __name__ == "__main__":
    unittest.main()