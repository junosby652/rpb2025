"""Read-only OS, WSL, NVIDIA, and Python diagnostics; run inside the venv."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = dict(platform=platform.platform(), kernel=platform.release(),
                  os_release=Path('/etc/os-release').read_text(),
                  python=sys.version, executable=sys.executable, in_venv=sys.prefix != sys.base_prefix,
                  wsl_kernel="microsoft" in platform.release().lower(),
                  windows_interop_registered=Path('/proc/sys/fs/binfmt_misc/WSLInterop').exists(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  nvidia=command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
                  nvidia_full=command(['nvidia-smi']),
                  wsl_version=command(['wsl.exe', '--version']),
                  windows_os=command(['powershell.exe', '-NoProfile', '-Command',
                      'Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber']))
    report['packages'] = {}
    for package in ('torch', 'mujoco', 'mujoco-warp', 'warp-lang', 'stable-baselines3', 'numpy'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None
    try:
        import torch
        report['torch_cuda_available'] = torch.cuda.is_available()
        report['torch_cuda_build'] = torch.version.cuda
        import warp as wp
        wp.init()
        report['warp_cuda_devices'] = [str(d) for d in wp.get_cuda_devices()]
    except Exception as exc:
        report['device_probe_error'] = repr(exc)
    result = json.dumps(report, indent=2, ensure_ascii=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + '\n')


if __name__ == '__main__':
    main()
