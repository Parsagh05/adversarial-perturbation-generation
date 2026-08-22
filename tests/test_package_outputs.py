from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from package_full_outputs import package_full_outputs


class FullOutputPackageTests(unittest.TestCase):
    def test_packages_setup_tree_without_nesting_existing_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_base = Path(directory)
            setup = output_base / "setups" / "frozen_prompt" / "steps500_eps2"
            setup.mkdir(parents=True)
            (setup / "attack_manifest.csv").write_text("scope\n", encoding="utf-8")
            (setup / "existing_scope.zip").write_bytes(b"already packaged")

            archive_path = package_full_outputs(output_base)

            self.assertEqual(archive_path, output_base / "full_outputs.zip")
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["setups/frozen_prompt/steps500_eps2/attack_manifest.csv"],
                )


if __name__ == "__main__":
    unittest.main()
