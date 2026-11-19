"""Run the local upload/process/download MOT application."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mot.app.jobs import JobManager
from mot.app.web import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--jobs-dir", type=Path, default=ROOT / "outputs/web_jobs")
    args = parser.parse_args()
    manager = JobManager(args.jobs_dir)
    server = make_server(args.host, args.port, manager)
    print(f"MOT app: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
