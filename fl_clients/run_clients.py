"""
run_clients.py  –  Launch all 10 FL clients in parallel via subprocess.

Usage:
    python fl_clients/run_clients.py                           # default: 10 clients, server 127.0.0.1:8080
    python fl_clients/run_clients.py --num-clients 5           # only launch clients 0-4
    python fl_clients/run_clients.py --server-address host:port

Press Ctrl+C to gracefully terminate all running client processes.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

NUM_CLIENTS_DEFAULT = 10
CLIENT_SCRIPT = Path(__file__).resolve().parent / "client.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn multiple FL client processes")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=NUM_CLIENTS_DEFAULT,
        help=f"Number of clients to launch (default: {NUM_CLIENTS_DEFAULT})",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Flower server address (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each client (offline test, no server needed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num = args.num_clients

    if num < 1 or num > NUM_CLIENTS_DEFAULT:
        print(f"[ERROR] --num-clients must be between 1 and {NUM_CLIENTS_DEFAULT}")
        sys.exit(1)

    if not CLIENT_SCRIPT.exists():
        print(f"[ERROR] client.py not found at {CLIENT_SCRIPT}")
        sys.exit(1)

    processes: list[subprocess.Popen] = []

    print(f"[LAUNCHER] Starting {num} FL client(s) ...")
    print(f"[LAUNCHER] Server address: {args.server_address}")
    print(f"[LAUNCHER] Dry-run mode: {args.dry_run}")
    print("-" * 50)

    try:
        for idx in range(num):
            cmd = [
                sys.executable,
                str(CLIENT_SCRIPT),
                "--client-index", str(idx),
                "--server-address", args.server_address,
            ]
            if args.dry_run:
                cmd.append("--dry-run")

            proc = subprocess.Popen(cmd)
            processes.append(proc)
            print(f"[LAUNCHER] Client {idx} (house{idx + 1}) started  |  PID={proc.pid}")

            # Small delay to avoid thundering herd on data loading
            if idx < num - 1:
                time.sleep(0.5)

        print("-" * 50)
        print(f"[LAUNCHER] All {num} clients launched. Press Ctrl+C to stop.")

        # Wait for all child processes to finish
        for proc in processes:
            proc.wait()

        print("[LAUNCHER] All clients have finished.")

    except KeyboardInterrupt:
        print("\n[LAUNCHER] Ctrl+C received – terminating all clients ...")
        for proc in processes:
            if proc.poll() is None:  # still running
                proc.terminate()

        # Give children a moment to exit cleanly
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print("[LAUNCHER] All clients terminated. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
