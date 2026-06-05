import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = REPO_ROOT / "scripts" / "reports" / "export_jstars_safe_figures.py"


def main():
    cmd = [sys.executable, str(TARGET_SCRIPT), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
