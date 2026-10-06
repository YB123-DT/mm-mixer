"""Materialize a minimally patched official ConFilMER checkout; no downloads/install."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
src=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for f in src.glob('*.py'):
 s=f.read_text()
 for dead in ['import ipdb\n','import clip\n','from clip import load\n','from torch_geometric.nn.pool.topk_pool import topk\n','from utils import get_args, set_manualSeed, image_transform, WinoLoss, CLIPLoss, MarginLoss\n']:
  s=s.replace(dead,'')
 if f.name=='HypergraphConv.py':
  s=s.replace('from torch_scatter import scatter_add','from torch_geometric.utils import scatter\ndef scatter_add(src, index, dim=0, dim_size=None):\n    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")')
  s=s.replace("self.flow = 'target_to_source'\n        out = self.propagate(hyperedge_index, x=out", "self.flow = 'source_to_target'\n        out = self.propagate(hyperedge_index.flip(0), x=out")
 if f.name=='train_our.py':
  s=s.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0"','assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Set an approved physical GPU explicitly"')
  s=s.replace('import numpy as np, argparse, time, pickle, random','import numpy as np, argparse, time, pickle, random\nimport json\nfrom pathlib import Path')
  s=s.replace('    parser = argparse.ArgumentParser()','    parser = argparse.ArgumentParser()\n    parser.add_argument("--seed", type=int, required=True)\n    parser.add_argument("--output-dir", required=True)\n    parser.add_argument("--smoke-batches", type=int, default=0)')
  s=s.replace('def seed_everything(seed=seed):','def seed_everything(seed=None):\n    seed = globals()["seed"] if seed is None else seed')
  s=s.replace('    args = parser.parse_args()','    args = parser.parse_args()\n    seed = args.seed\n    output_dir = Path(args.output_dir)\n    output_dir.mkdir(parents=True, exist_ok=True)\n    assert not any((output_dir / n).exists() for n in ("config.json", "result.json", "best.pt", "completed.json")), "Refusing to overwrite existing run"\n    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))')
  s=s.replace('    clip_model, preprocess = load("ViT-B/32", jit=False)\n    clip_model = clip_model.cuda()','    clip_model = nn.Identity()  # official forward never reads clip_model')
  s=s.replace('    for data in dataloader:', '    for batch_index, data in enumerate(dataloader):\n        if args.smoke_batches and batch_index >= args.smoke_batches:\n            break')
  s=s.replace('if preds!=[]:', 'if len(preds):')
  marker='            best_label, best_pred = test_label, test_pred\n'
  replacement=marker+'''            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": e + 1, "seed": seed, "config": vars(args)}, output_dir / "best.pt")
            np.savez_compressed(output_dir / "predictions.npz", y_true=test_label, y_pred=test_pred)
            (output_dir / "best.json").write_text(json.dumps({"epoch": e + 1, "accuracy": float(accuracy_score(test_label, test_pred)*100), "weighted_f1": float(f1_score(test_label, test_pred, average="weighted")*100), "macro_f1": float(f1_score(test_label, test_pred, average="macro")*100), "seed": seed, "selection": "strict_peak_test_wf1", "smoke_only": bool(args.smoke_batches)}, indent=2))
'''
  assert marker in s;s=s.replace(marker,replacement)
  s=s.replace("        start_time = time.time()", "        start_time = time.time()\n        if cuda: torch.cuda.reset_peak_memory_stats()")
  s=s.replace("        if (e+1)%10 == 0:", '        with (output_dir / "epochs.jsonl").open("a") as stream:\n            stream.write(json.dumps({"epoch": e+1, "seconds": time.time()-start_time, "train_loss": train_loss, "train_accuracy": train_acc, "train_weighted_f1": train_fscore, "test_loss": test_loss, "test_accuracy": test_acc, "test_weighted_f1": test_fscore, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0, "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0})+"\\n")\n        if (e+1)%10 == 0:')
  s += '\n    saved = torch.load(output_dir / "best.pt", map_location="cuda" if cuda else "cpu")\n    model.load_state_dict(saved["model"], strict=True)\n    reloaded = train_or_eval_graph_model(model, clip_model, loss_function, test_loader, saved["epoch"]-1, cuda, args.modals, dataset=args.Dataset)\n    archived = np.load(output_dir / "predictions.npz")\n    assert np.array_equal(reloaded[2], archived["y_true"]) and np.array_equal(reloaded[3], archived["y_pred"]), "Strict reload predictions differ"\n'
  s += '\n    best_result = json.loads((output_dir / "best.json").read_text())\n    (output_dir / "result.json").write_text(json.dumps({"status": "smoke_completed" if args.smoke_batches else "completed", "model": "ConFilMER", "dataset": args.Dataset.lower(), "seed": seed, "epochs_completed": n_epochs, "selection": "strict_peak_test_wf1", "selected_epoch": best_result["epoch"], "selection_metric_round_decimals": 2, "strict_reload_predictions_match": True, "smoke_only": bool(args.smoke_batches), "test": {k: best_result[k] for k in ("accuracy", "weighted_f1", "macro_f1")}}, indent=2))\n'
  s += '\n    (output_dir / "completed.json").write_text(json.dumps({"status": "smoke_completed" if args.smoke_batches else "completed", "epochs": n_epochs, "smoke_only": bool(args.smoke_batches)}))\n'
 (out/f.name).write_text(s)
print(out)
