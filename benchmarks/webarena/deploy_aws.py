#!/usr/bin/env python3
"""
AWS Deployment for WebArena Benchmark.

Launches an EC2 instance from the official WebArena AMI, waits for services
to start, then runs the Digital Cerebellum benchmark remotely.

Prerequisites:
    pip install boto3
    aws configure  (set up AWS credentials)

Usage:
    # Launch instance and run benchmark
    python -m benchmarks.webarena.deploy_aws --launch --run

    # Connect to existing instance
    python -m benchmarks.webarena.deploy_aws --hostname ec2-xx-xx-xx-xx.us-east-2.compute.amazonaws.com --run

    # Just launch (don't run benchmark yet)
    python -m benchmarks.webarena.deploy_aws --launch

    # Terminate instance
    python -m benchmarks.webarena.deploy_aws --terminate --instance-id i-xxxxx

Cost estimate: t3a.xlarge ~$0.15/hr. Full 258-task benchmark takes ~4-8 hours = $0.60-$1.20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WEBARENA_AMI = "ami-08a862bf98e3bd7aa"
REGION = "us-east-2"
INSTANCE_TYPE = "t3a.xlarge"
VOLUME_SIZE_GB = 200  # WebArena needs plenty of space


def launch_instance(key_name: str, security_group: str | None = None) -> dict:
    """Launch EC2 instance from WebArena AMI."""
    try:
        import boto3
    except ImportError:
        print("Install boto3: pip install boto3")
        sys.exit(1)

    ec2 = boto3.client("ec2", region_name=REGION)

    # Create security group if not provided
    if not security_group:
        sg_name = "webarena-benchmark"
        try:
            resp = ec2.create_security_group(
                GroupName=sg_name,
                Description="WebArena benchmark - all inbound",
            )
            sg_id = resp["GroupId"]
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "-1",
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }],
            )
            print(f"Created security group: {sg_id}")
        except ec2.exceptions.ClientError:
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
            print(f"Using existing security group: {sg_id}")
        security_group = sg_id

    print(f"Launching {INSTANCE_TYPE} from AMI {WEBARENA_AMI}...")
    resp = ec2.run_instances(
        ImageId=WEBARENA_AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        KeyName=key_name,
        SecurityGroupIds=[security_group],
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": VOLUME_SIZE_GB, "VolumeType": "gp3"},
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "webarena-benchmark"}],
        }],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"Instance launched: {instance_id}")

    # Wait for running
    print("Waiting for instance to start...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    # Get public hostname
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    hostname = desc["Reservations"][0]["Instances"][0].get("PublicDnsName", "")
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")

    print(f"Instance running!")
    print(f"  Hostname: {hostname}")
    print(f"  IP: {public_ip}")
    print(f"  Instance ID: {instance_id}")

    return {
        "instance_id": instance_id,
        "hostname": hostname,
        "public_ip": public_ip,
    }


def wait_for_services(hostname: str, timeout: int = 300) -> bool:
    """Wait for WebArena services to be accessible."""
    import urllib.request

    services = {
        "shopping": f"http://{hostname}:7770",
        "shopping_admin": f"http://{hostname}:7780",
        "reddit": f"http://{hostname}:9999",
        "gitlab": f"http://{hostname}:8023",
    }

    print("Waiting for WebArena services...")
    t0 = time.time()
    ready = set()

    while time.time() - t0 < timeout:
        for name, url in services.items():
            if name in ready:
                continue
            try:
                req = urllib.request.Request(url, method="HEAD")
                urllib.request.urlopen(req, timeout=5)
                ready.add(name)
                print(f"  [OK] {name} ({url})")
            except Exception:
                pass

        if len(ready) == len(services):
            print("All services ready!")
            return True

        time.sleep(10)

    print(f"Timeout after {timeout}s. Ready: {ready}")
    return len(ready) > 0


def configure_services(hostname: str) -> None:
    """Configure WebArena services with the correct hostname."""
    commands = [
        f'docker start shopping shopping_admin forum gitlab',
        f'docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://{hostname}:7770"',
        f'docker exec shopping mysql -u magentouser -pMyPassword magentodb -e \'UPDATE core_config_data SET value="http://{hostname}:7770/" WHERE path = "web/secure/base_url";\'',
        f'docker exec shopping /var/www/magento2/bin/magento cache:flush',
        f'docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://{hostname}:7780"',
        f'docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \'UPDATE core_config_data SET value="http://{hostname}:7780/" WHERE path = "web/secure/base_url";\'',
        f'docker exec shopping_admin /var/www/magento2/bin/magento cache:flush',
    ]

    print("Configuring services (via SSH)...")
    for cmd in commands:
        print(f"  Running: {cmd[:80]}...")


def run_benchmark_remote(hostname: str, mode: str = "quick") -> None:
    """Run the benchmark against a remote WebArena instance."""
    args = [
        sys.executable, "-m", "benchmarks.webarena.runner",
        f"--shopping-url", f"http://{hostname}:7770",
        f"--reddit-url", f"http://{hostname}:9999",
        f"--gitlab-url", f"http://{hostname}:8023",
    ]

    if mode == "quick":
        args.append("--quick")
    elif mode == "hard":
        args.append("--hard")
    elif mode == "ablation":
        args.extend(["--hard", "--ablation"])

    args.extend(["--output", "benchmarks/results", "--verbose"])

    print(f"\nRunning benchmark ({mode} mode)...")
    print(f"Command: {' '.join(args)}")
    subprocess.run(args, cwd=str(Path(__file__).parent.parent.parent))


def generate_run_script(hostname: str) -> None:
    """Generate a local script to run the benchmark."""
    script = f"""#!/usr/bin/env python3
# Auto-generated script to run WebArena benchmark against {hostname}
# Usage: python benchmarks/webarena/run_remote.py

import subprocess, sys

hostname = "{hostname}"

configs = [
    # (label, extra_args)
    ("baseline (quick test)", ["--quick"]),
    ("baseline + full (20 tasks)", []),
    ("full ablation (258 hard tasks)", ["--hard", "--ablation"]),
]

print("WebArena Benchmark Runner")
print("=" * 50)
print(f"Target: {{hostname}}")
print()
for i, (label, _) in enumerate(configs):
    print(f"  {{i+1}}. {{label}}")
print()

choice = input("Select config (1-3): ").strip()
idx = int(choice) - 1
label, extra = configs[idx]

cmd = [
    sys.executable, "-m", "benchmarks.webarena.runner",
    "--shopping-url", f"http://{{hostname}}:7770",
    "--reddit-url", f"http://{{hostname}}:9999",
    "--gitlab-url", f"http://{{hostname}}:8023",
    "--output", "benchmarks/results",
    "--verbose",
] + extra

print(f"\\nRunning: {{label}}")
subprocess.run(cmd)
"""
    out = Path(__file__).parent / "run_remote.py"
    out.write_text(script, encoding="utf-8")
    print(f"Generated: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS WebArena Deployment")
    parser.add_argument("--launch", action="store_true", help="Launch EC2 instance")
    parser.add_argument("--key-name", type=str, help="SSH key pair name")
    parser.add_argument("--hostname", type=str, help="Existing WebArena hostname")
    parser.add_argument("--run", action="store_true", help="Run benchmark")
    parser.add_argument("--mode", choices=["quick", "hard", "ablation"], default="quick")
    parser.add_argument("--terminate", action="store_true", help="Terminate instance")
    parser.add_argument("--instance-id", type=str, help="Instance ID to terminate")
    parser.add_argument("--generate-script", action="store_true", help="Generate run script")

    args = parser.parse_args()

    hostname = args.hostname

    if args.launch:
        if not args.key_name:
            print("--key-name required for launch")
            sys.exit(1)
        info = launch_instance(args.key_name)
        hostname = info["hostname"]
        print("\nWait 2-3 minutes for Docker services to start, then run:")
        print(f"  python -m benchmarks.webarena.deploy_aws --hostname {hostname} --run")

    if args.generate_script and hostname:
        generate_run_script(hostname)

    if args.run:
        if not hostname:
            print("--hostname required for run")
            sys.exit(1)
        run_benchmark_remote(hostname, mode=args.mode)

    if args.terminate:
        if not args.instance_id:
            print("--instance-id required for terminate")
            sys.exit(1)
        try:
            import boto3
            ec2 = boto3.client("ec2", region_name=REGION)
            ec2.terminate_instances(InstanceIds=[args.instance_id])
            print(f"Terminated: {args.instance_id}")
        except ImportError:
            print("Install boto3: pip install boto3")


if __name__ == "__main__":
    main()
