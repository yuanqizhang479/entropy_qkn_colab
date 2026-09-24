"""Record hardware/software provenance and check the installed CUDA build."""
import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--out', default='outputs/environment.json')
p.add_argument('--require-cuda', action='store_true')
a = p.parse_args()
if sys.version_info < (3, 11):
    raise SystemExit('Use Python >=3.11; local audited reference used Python3.12')
if a.require_cuda and not torch.cuda.is_available():
    raise SystemExit('GPU unavailable. In Colab choose Runtime > Change runtime type > GPU.')
versions = {}
for name in ['torch','numpy','scipy','matplotlib','tiktoken','pyarrow','pytest']:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        raise SystemExit(f'Missing {name}: install requirements.txt first')
try:
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
except (FileNotFoundError, subprocess.CalledProcessError):
    git_sha, dirty = None, None
info = {'python':sys.version,'platform':platform.platform(),'versions':versions,
        'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,
        'git_commit':git_sha,'git_dirty':dirty,
        'gpus':[{'name':torch.cuda.get_device_name(i),
                 'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory}
                for i in range(torch.cuda.device_count())]}
if torch.cuda.is_available():
    x=torch.randn(8,8,device='cuda',requires_grad=True)
    (x@x).sum().backward()
    torch.cuda.synchronize()
    info['cuda_forward_backward_smoke']='passed'
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
