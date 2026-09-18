import subprocess
import sys
from pathlib import Path


def test_src_and_scripts_import_from_any_cwd(tmp_path: Path) -> None:
    # Given an interpreter started outside the repository.
    result = subprocess.run(
        [sys.executable, "-c", "import src.models, scripts.hipodit_multiseed_summary"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    # Then both packages resolve without any sys.path manipulation.
    assert result.returncode == 0, result.stderr
