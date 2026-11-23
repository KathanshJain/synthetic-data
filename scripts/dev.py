"""
Start BOTH servers for local development:
  - API   -> http://localhost:8000  (FastAPI, docs at /docs)
  - Web   -> http://localhost:5173  (Vite dev server for frontend/web)

Usage:  python scripts/dev.py     (or run.bat on Windows)
Ctrl+C stops both.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def main():
    procs = []
    try:
        procs.append(subprocess.Popen(
            [PY, "-m", "uvicorn", "api.main:app", "--port", "8000", "--reload"],
            cwd=ROOT))
        print("API  -> http://localhost:8000  (docs at /docs)")
        web = ROOT / "frontend" / "web"
        procs.append(subprocess.Popen(
            ["npm", "run", "dev"], cwd=web,
            shell=(sys.platform == "win32")))
        print("Web  -> http://localhost:5173")
        while True:
            time.sleep(5)
            for proc in procs:
                if proc.poll() is not None:
                    print(f"a server exited with {proc.returncode}; "
                          "shutting down")
                    sys.exit(proc.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
