#!/usr/bin/env python3
"""Prepare an isolated sensitivity snapshot from the original revision commit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

BASE = '1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def write_manifest(root):
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'
              and p.name != 'snapshot.json' and '.pytest_cache' not in p.parts}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = {'base_commit': BASE, 'experiment': 'projection_view_amm_capacity_20261009',
                'sha256': digest, 'file_sha256': hashes,
                'patch_sha256': hashlib.sha256((HERE / 'controls.patch').read_bytes()).hexdigest(),
                'change_rule': 'Planned runs only widen both AMM block FFNs to 3072 or 6144 at S=6,D=256,depth=2; unused joint-deletion control inherited from prior verified patch' }
    (root / 'snapshot.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'root': str(root), 'sha256': digest, 'files': len(hashes)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=REPO / 'outputs/amm_capacity_20261009/code')
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    root = args.destination.resolve()
    if not args.manifest_only:
        if root.exists() and any(root.iterdir()):
            raise SystemExit('Refusing to overwrite a nonempty snapshot')
        root.mkdir(parents=True, exist_ok=True)
        raw = subprocess.check_output(['git', '-C', str(REPO), 'archive', BASE])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(root)
        subprocess.run(['git', 'apply', '--unsafe-paths', '--directory=' + str(root), str(HERE / 'controls.patch')], check=True, cwd='/')
    write_manifest(root)


if __name__ == '__main__':
    main()
