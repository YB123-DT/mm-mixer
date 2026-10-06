#!/usr/bin/env python3
"""Run a fixed list of baseline commands on explicitly allowed biggpu devices.

No retries, model changes, package installation, or automatic result selection.
Keep the controller in tmux. Restart only after inspecting any interrupted job.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_state(allowed):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for line in output.splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        if uuid in allowed:
            if index == '4':
                raise RuntimeError('biggpu GPU 4 must never be used')
            devices[uuid] = {'index': int(index), 'free_mb': int(free)}
    if set(devices) != set(allowed):
        raise RuntimeError('GPU whitelist identity mismatch')
    return devices


def verify_sources(job):
    for filename, expected in job['source_sha256'].items():
        actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'frozen source changed: {filename}')


def check_result(job):
    path = Path(job['output']) / 'result.json'
    data = json.loads(path.read_text())
    if data['status'] != 'completed':
        raise ValueError('result is not completed')
    if data.get('smoke_only', False) or data.get('smoke', False):
        raise ValueError('smoke result is not a formal run')
    if data.get('epochs_completed', 0) < job.get('min_epochs', 2):
        raise ValueError('formal training budget not reached')
    for key in ('model', 'dataset', 'seed'):
        if data[key] != job[key]:
            raise ValueError(f'result identity mismatch: {key}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = plan['jobs']
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate job IDs')
    if len({j['output'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate output paths')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    state = {'plan_sha256': plan_hash, 'jobs': {}}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state['plan_sha256'] != plan_hash:
            raise ValueError('plan changed')
        if any(v['status'] == 'running' for v in state['jobs'].values()):
            raise RuntimeError('inspect interrupted controller jobs before restart')
    active = {}
    try:
        while True:
            for identity, (process, log, job, gpu) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                record = state['jobs'][identity]
                record.update(returncode=rc, ended_at=time.time(), status='failed')
                if rc == 0:
                    try:
                        check_result(job)
                        record['status'] = 'completed'
                    except Exception as exc:
                        record['error'] = str(exc)
                del active[identity]
                save(args.state, state)
            pending = [j for j in jobs if j['id'] not in state['jobs']]
            if not pending and not active:
                break
            devices = gpu_state(plan['gpu_uuids'])
            for job in pending:
                gpu = next((uuid for uuid, info in devices.items()
                            if sum(x[3] == uuid for x in active.values()) < plan['per_gpu']
                            and info['free_mb'] >= job['required_free_mb']), None)
                if gpu is None:
                    continue
                verify_sources(job)
                out = Path(job['output'])
                if out.exists():
                    raise RuntimeError(f'run directory already exists: {out}')
                out.mkdir(parents=True)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                env['PYTHONHASHSEED'] = str(job['seed'])
                # multiprocessing AF_UNIX sockets have a ~108-byte path limit.
                temporary = Path('/data2/yb/tmp/mmbl') / hashlib.sha256(str(out).encode()).hexdigest()[:12]
                temporary.mkdir(parents=True, exist_ok=True)
                env.update(TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary))
                env.update(job.get('environment', {}))
                if env['CUDA_VISIBLE_DEVICES'] != gpu:
                    raise ValueError('job must not override GPU allocation')
                record = {'status': 'running', 'started_at': time.time(),
                          'gpu_uuid': gpu, 'gpu_index': devices[gpu]['index'],
                          'command': job['command'], 'cwd': job['cwd']}
                save(out / 'launch.json', record)
                log = (out / 'console.log').open('w')
                process = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = process.pid
                save(out / 'launch.json', record)
                state['jobs'][job['id']] = record
                active[job['id']] = (process, log, job, gpu)
                save(args.state, state)
                # Observe actual allocations before admitting another job.
                time.sleep(plan.get('launch_interval_seconds', 20))
                devices = gpu_state(plan['gpu_uuids'])
            time.sleep(10)
        state['status'] = ('completed' if all(v['status'] == 'completed'
                          for v in state['jobs'].values()) else 'finished_with_failures')
        state['ended_at'] = time.time()
        save(args.state, state)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
