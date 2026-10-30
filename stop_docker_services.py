import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    if not shutil.which("docker"):
        print("Docker CLI was not found. Install/start Docker Desktop or Docker Engine.", file=sys.stderr)
        return 1

    try:
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=ROOT,
            check=True,
        )
        print("Docker Compose services stopped. Named volumes were preserved.")
        return 0
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Could not stop Docker services: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
