"""One-GPU serial, resumable queue with process locks; no implicit launch."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

from .formal_runner import ROOT, atomic_json, verify_run_artifacts


def partition_queue(ids, partitions: int, index: int):
    return list(ids)[index::partitions]


def run_is_complete(root: Path, eid: str):
    return verify_run_artifacts(root,eid)


def pending_experiments(root: Path, ids):
    pending=[]
    for eid in ids:
        if run_is_complete(root,eid): continue
        pending.append(eid)
    return pending


def run_queue(root: Path, ids, gpu: int):
    import fcntl
    root=Path(root); queue_dir=root/"queues"; queue_dir.mkdir(parents=True,exist_ok=True)
    lock=(queue_dir/f"gpu{gpu}.lock").open("a+")
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError(f"GPU {gpu} queue already running") from exc
    try:
        atomic_json(queue_dir/f"gpu{gpu}.json",{"state":"running","gpu":gpu,"pid":os.getpid(),"ids":ids})
        for eid in pending_experiments(root,ids):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            command=[sys.executable,"-m","ablation48.formal_runner","--config",str(root/"configs"/f"{eid}.json"),"--output-root",str(root),"--execute"]
            log=root/"runs"/eid/"train.log"; log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("a") as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(queue_dir/f"gpu{gpu}.json",{"state":"failed","gpu":gpu,"experiment_id":eid,"exit_code":result.returncode})
                return result.returncode
        atomic_json(queue_dir/f"gpu{gpu}.json",{"state":"complete","gpu":gpu,"ids":ids})
        return 0
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",required=True)
    parser.add_argument("--gpu",type=int,required=True); parser.add_argument("--ids",nargs="+")
    parser.add_argument("--partitions",type=int); parser.add_argument("--partition-index",type=int)
    parser.add_argument("--execute",action="store_true"); args=parser.parse_args(argv)
    if args.ids:
        ids=args.ids
        if args.partitions is not None: ids=partition_queue(ids,args.partitions,args.partition_index)
    else:
        utterance=[f"{i:02d}" for i in range(1,47) if i not in {41,42,43,44}]
        ids=partition_queue(utterance,args.partitions,args.partition_index) if args.partitions is not None else utterance
        if args.partitions is None or args.partition_index==0:
            ids=["DIALOGUE_NULL",*ids,"41","42","43","44"]
    if not args.execute:
        print(json.dumps({"dry_run":True,"gpu":args.gpu,"pending":pending_experiments(Path(args.output_root),ids)},indent=2)); return 0
    return run_queue(Path(args.output_root),ids,args.gpu)


if __name__=="__main__": raise SystemExit(main())
