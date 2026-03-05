#!/usr/bin/env python3
"""
WebArena Docker Environment Setup.

Sets up the WebArena web services via Docker Compose.
Run this AFTER Docker Desktop is running.

Usage:
    python -m benchmarks.webarena.setup_docker --check
    python -m benchmarks.webarena.setup_docker --start
    python -m benchmarks.webarena.setup_docker --stop

Prerequisites:
    1. Start Docker Desktop manually (GUI)
    2. Ensure Docker daemon is running: `docker info`
    3. Ensure E: drive has 15GB+ free space
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

COMPOSE_DIR = Path(__file__).parent / "docker"
COMPOSE_FILE = COMPOSE_DIR / "docker-compose.yml"

IMAGE_DOWNLOADS = {
    "shopping": {
        "tar": "shopping_final_0712.tar",
        "urls": [
            "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar",
            "https://archive.org/download/webarena-env-shopping-image/shopping_final_0712.tar",
        ],
        "docker_name": "shopping_final_0712",
        "container": "shopping",
        "port": "7770:80",
    },
    "shopping_admin": {
        "tar": "shopping_admin_final_0719.tar",
        "urls": [
            "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar",
            "https://archive.org/download/webarena-env-shopping-admin-image/shopping_admin_final_0719.tar",
        ],
        "docker_name": "shopping_admin_final_0719",
        "container": "shopping_admin",
        "port": "7780:80",
    },
    "reddit": {
        "tar": "postmill-populated-exposed-withimg.tar",
        "urls": [
            "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar",
            "https://archive.org/download/webarena-env-forum-image/postmill-populated-exposed-withimg.tar",
        ],
        "docker_name": "postmill-populated-exposed-withimg",
        "container": "forum",
        "port": "9999:80",
    },
    "gitlab": {
        "tar": "gitlab-populated-final-port8023.tar",
        "urls": [
            "http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar",
            "https://archive.org/download/webarena-env-gitlab-image/gitlab-populated-final-port8023.tar",
        ],
        "docker_name": "gitlab-populated-final-port8023",
        "container": "gitlab",
        "port": "8023:8023",
    },
}

COMPOSE_CONTENT = """# NOTE: WebArena images are NOT on Docker Hub / GHCR.
# They must be downloaded as tar files and loaded with `docker load`.
# Use: python -m benchmarks.webarena.setup_docker --start
# Or use AWS AMI: python -m benchmarks.webarena.deploy_aws --launch
#
# Each image is ~40GB. Total: ~160GB. Use AWS EC2 for best experience.
"""


def check_docker() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print("[OK] Docker daemon is running")
            return True
        else:
            print("[FAIL] Docker daemon not responding")
            print("  Start Docker Desktop manually, then retry.")
            return False
    except FileNotFoundError:
        print("[FAIL] docker command not found")
        return False
    except subprocess.TimeoutExpired:
        print("[FAIL] Docker daemon timed out")
        return False


def check_disk_space() -> bool:
    """Check if there's enough disk space."""
    usage = shutil.disk_usage("E:\\")
    free_gb = usage.free / (1024**3)
    print(f"  E: drive free: {free_gb:.1f} GB")
    if free_gb < 10:
        print("[WARN] Less than 10GB free on E: — may not be enough")
        return False
    print("[OK] Sufficient disk space")
    return True


def setup_compose() -> None:
    """Create the docker-compose.yml file."""
    COMPOSE_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSE_FILE.write_text(COMPOSE_CONTENT, encoding="utf-8")
    print(f"[OK] Created {COMPOSE_FILE}")


def start_services() -> None:
    """Start WebArena Docker services."""
    if not check_docker():
        sys.exit(1)

    print("\n[INFO] WebArena images are ~40GB each and must be downloaded as tar files.")
    print("       Recommended: use AWS AMI instead (python -m benchmarks.webarena.deploy_aws)")
    print()

    for name, info in IMAGE_DOWNLOADS.items():
        tar_path = COMPOSE_DIR / info["tar"]
        if tar_path.exists():
            print(f"\n--- Loading {name} from {tar_path} ---")
            subprocess.run(["docker", "load", "--input", str(tar_path)], check=True)
            cmd = f'docker run --name {info["container"]} -p {info["port"]} -d {info["docker_name"]}'
            print(f"  Running: {cmd}")
            subprocess.run(cmd.split(), check=True)
            print(f"  [OK] {name} started")
        else:
            print(f"\n[SKIP] {name}: tar not found at {tar_path}")
            print(f"  Download from: {info['urls'][0]}")

    print("\nTo verify: python -m benchmarks.webarena.setup_docker --check")


def stop_services() -> None:
    """Stop WebArena Docker services."""
    if COMPOSE_FILE.exists():
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
            check=True,
        )
        print("[OK] Services stopped")
    else:
        print("No docker-compose.yml found")


def check_services() -> None:
    """Check status of all components."""
    print("=== WebArena Environment Check ===\n")

    ok = check_docker()
    if not ok:
        return

    check_disk_space()

    # Check running containers
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
        capture_output=True, text=True,
    )
    webarena = [
        line for line in result.stdout.strip().split("\n")
        if line and "webarena" in line
    ]

    if webarena:
        print(f"\n[OK] {len(webarena)} WebArena container(s) running:")
        for line in webarena:
            print(f"  {line}")
    else:
        print("\n[INFO] No WebArena containers running")
        print("  Run: python -m benchmarks.webarena.setup_docker --start")


def main() -> None:
    parser = argparse.ArgumentParser(description="WebArena Docker Setup")
    parser.add_argument("--check", action="store_true", help="Check environment status")
    parser.add_argument("--start", action="store_true", help="Start WebArena services")
    parser.add_argument("--stop", action="store_true", help="Stop WebArena services")
    args = parser.parse_args()

    if args.start:
        start_services()
    elif args.stop:
        stop_services()
    else:
        check_services()


if __name__ == "__main__":
    main()
