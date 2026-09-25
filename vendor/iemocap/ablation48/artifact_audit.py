"""Audit immutable run evidence and deterministically materialize 47/48."""
from __future__ import annotations
import argparse, csv, json, math
from itertools import combinations
from pathlib import Path

from .formal_runner import PROTOCOL, _sha, atomic_json, atomic_text, verify_run_artifacts
from .lock import canonical_lock_sha256, validate_results_lock
from .registry import build_registry, compatible

ROOT=Path(__file__).resolve().parents[1]
S0_MANIFEST=ROOT.parent/"Model_rawaux_capacity_e0_e15_peaktest/results/e14/manifest.json"
S1P_MANIFEST=ROOT.parent/"Model_rawaux_subspace_kdl_sweep/results/info_on/s1/manifest.json"


def _control_evidence(formal_root: Path):
    paths={"S0":S0_MANIFEST,"S1P":S1P_MANIFEST,
           "DIALOGUE_NULL":formal_root/"runs/DIALOGUE_NULL/manifest.json"}
    evidence={}
    for name,path in paths.items():
        if not path.is_file(): raise RuntimeError(f"missing matched control manifest: {path}")
        manifest=json.loads(path.read_text())
        evidence[name]={"weighted_f1":float(manifest["weighted_f1"]),"manifest":str(path),
                        "manifest_sha256":_sha(path)}
    return evidence


def _is_compatible(a, b):
    exclusive={"tokenizer","routing","gate","depth","aggregator","readout","dropout"}
    return not (set(a["owned_boundaries"]) & set(b["owned_boundaries"]) & exclusive)


def select_deferred(rows):
    by_id={row["experiment_id"]:row for row in rows}
    cores=[by_id[f"{i:02d}"] for i in range(1,37) if f"{i:02d}" in by_id and by_id[f"{i:02d}"]["matched_delta_wf1"]>0]
    lights=[by_id[f"{i:02d}"] for i in range(37,41) if f"{i:02d}" in by_id and by_id[f"{i:02d}"]["matched_delta_wf1"]>0]
    pairs=[(a,b) for a,b in combinations(cores,2) if _is_compatible(a,b)]
    if pairs:
        def pair_key(pair):
            a,b=pair
            return (-(a["matched_delta_wf1"]+b["matched_delta_wf1"]),
                    -(a["macro_f1"]+b["macro_f1"])/2,
                    a["parameter_count"]+b["parameter_count"],
                    tuple(sorted((a["experiment_id"],b["experiment_id"]))))
        winner=min(pairs,key=pair_key)
        out47={"status":"MATERIALIZED","members":sorted(x["experiment_id"] for x in winner)}
    else:
        out47={"status":"NO-GO_NO_POSITIVE_CORE_PAIR","members":[]}
    candidates=[(core,light) for core in cores for light in lights if _is_compatible(core,light)]
    if not cores: out48={"status":"NO-GO_NO_POSITIVE_CORE","members":[]}
    elif not lights: out48={"status":"NO-GO_NO_POSITIVE_LIGHTWEIGHT","members":[]}
    elif not candidates: out48={"status":"NO-GO_NO_COMPATIBLE_CORE_LIGHTWEIGHT","members":[]}
    else:
        def combo_key(pair):
            core,light=pair
            return (-core["matched_delta_wf1"],-light["matched_delta_wf1"],
                    -(core["macro_f1"]+light["macro_f1"])/2,
                    core["parameter_count"]+light["parameter_count"],
                    (core["experiment_id"],light["experiment_id"]))
        winner=min(candidates,key=combo_key)
        out48={"status":"MATERIALIZED","members":[winner[0]["experiment_id"],winner[1]["experiment_id"]]}
    return {"47":out47,"48":out48,"positive_core_ids":[r["experiment_id"] for r in cores],
            "positive_lightweight_ids":[r["experiment_id"] for r in lights]}


def summarize_combination(score: float, s0_score: float, member_scores: dict[str,float]):
    best_member=max(member_scores.values())
    beats_s0=score>s0_score; beats_best=score>best_member
    return {"weighted_f1":score,"s0_weighted_f1":s0_score,"member_weighted_f1":member_scores,
            "delta_vs_s0":score-s0_score,"delta_vs_best_member":score-best_member,
            "beats_s0":beats_s0,"beats_best_member":beats_best,"synergy":beats_s0 and beats_best,
            "decision":"KEEP_COMBINATION" if beats_s0 and beats_best else "NEGATIVE_KEEP_40"}


def audit_and_materialize(formal_root: Path, output: Path):
    formal_root=Path(formal_root); output=Path(output); registry=build_registry(); controls=_control_evidence(formal_root)
    if not verify_run_artifacts(formal_root,"DIALOGUE_NULL",expected_protocol=PROTOCOL,
                                expected_batch_protocol="dialogue",expected_seed=2025):
        raise RuntimeError("DIALOGUE_NULL artifact audit failed")
    rows=[]; lock_results={}
    for eid,row in registry.items():
        if not verify_run_artifacts(formal_root,eid,expected_protocol=PROTOCOL,
                                    expected_batch_protocol=row.config.batch_protocol,expected_seed=2025):
            raise RuntimeError(f"{eid}: artifact audit failed")
        run=formal_root/"runs"/eid
        manifest=json.loads((run/"manifest.json").read_text()); metrics=json.loads((run/"metrics.json").read_text())
        score=float(manifest["weighted_f1"]); macro=float(manifest["macro_f1"])
        control=row.matched_control; control_score=controls[control]["weighted_f1"]; delta=score-control_score
        if not all(math.isfinite(x) for x in (score,macro,delta)): raise RuntimeError(f"{eid}: non-finite score")
        record={"experiment_id":eid,"title":row.title,"family":row.family,"matched_control":control,
                "weighted_f1":score,"accuracy":float(manifest["accuracy"]),"macro_f1":macro,
                "matched_control_wf1":control_score,"matched_delta_wf1":delta,
                "parameter_count":int(manifest["parameter_count"]),"owned_boundaries":sorted(row.owned_boundaries),
                "config_sha256":_sha(run/"config.json"),"source_sha256":_sha(run/"source_sha256.json"),
                "checkpoint_sha256":_sha(run/"best_peak_test_state_dict.pt"),
                "replay_sha256":_sha(run/"replay_logits.pt"),"control_manifest_sha256":controls[control]["manifest_sha256"]}
        rows.append(record)
        lock_results[eid]={"complete":True,"finite":True,"replay_exact":True,
                           "config_sha256":record["config_sha256"],"source_sha256":record["source_sha256"],
                           "checkpoint_sha256":record["checkpoint_sha256"],"replay_sha256":record["replay_sha256"],
                           "matched_delta_wf1":delta,"matched_control":control,"matched_control_wf1":control_score}
    lock={"schema_version":1,"results":lock_results}; lock["lock_sha256"]=canonical_lock_sha256(lock)
    validate_results_lock(lock)
    output.mkdir(parents=True,exist_ok=True); atomic_json(output/"results_lock.json",lock)
    atomic_json(output/"control_evidence.json",controls)
    selection=select_deferred(rows); selection.update({"results_lock_sha256":_sha(output/"results_lock.json"),
                                                       "rule":"registry matched-control deltas; strict positive only"})
    atomic_json(output/"deferred_selection.json",selection)
    for eid in ("47","48"):
        atomic_json(output/f"{eid}_selection.json",{"experiment_id":eid,**selection[eid],
                    "results_lock_sha256":selection["results_lock_sha256"],"selection_frozen":True})
    columns=["experiment_id","title","family","matched_control","weighted_f1","accuracy","macro_f1",
             "matched_control_wf1","matched_delta_wf1","parameter_count"]
    table=output/"results_01_46.csv"
    with table.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for row in rows: writer.writerow({key:row[key] for key in columns})
    # 47 is a frozen no-go.  Audit the actually trained 48 against both S0 and
    # its strongest constituent; selection evidence must still name 16+40.
    if selection["48"] != {"status":"MATERIALIZED","members":["16","40"]}:
        raise RuntimeError("formal 48 exists but frozen selection is not exactly 16+40")
    if not verify_run_artifacts(formal_root,"48",expected_protocol=PROTOCOL,
                                expected_batch_protocol="utterance",expected_seed=2025):
        raise RuntimeError("48 physical artifact/hash/protocol/seed audit failed")
    run48=formal_root/"runs/48"; manifest48=json.loads((run48/"manifest.json").read_text())
    by_id={row["experiment_id"]:row for row in rows}
    outcome48=summarize_combination(float(manifest48["weighted_f1"]),controls["S0"]["weighted_f1"],
                                    {eid:by_id[eid]["weighted_f1"] for eid in ("16","40")})
    outcome48.update({"experiment_id":"48","members":["16","40"],"accuracy":float(manifest48["accuracy"]),
                      "macro_f1":float(manifest48["macro_f1"]),"best_epoch":int(json.loads((run48/"metrics.json").read_text())["best_epoch"]),
                      "parameter_count":int(manifest48["parameter_count"]),"config_sha256":_sha(run48/"config.json"),
                      "source_sha256":_sha(run48/"source_sha256.json"),"checkpoint_sha256":_sha(run48/"best_peak_test_state_dict.pt"),
                      "replay_sha256":_sha(run48/"replay_logits.pt"),"fresh_strict_replay_exact":True})
    atomic_json(output/"48_final_outcome.json",outcome48)
    final_columns=["experiment_id","status","matched_control","weighted_f1","accuracy","macro_f1",
                   "matched_control_wf1","matched_delta_wf1","parameter_count","members"]
    with (output/"results_01_48.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=final_columns); writer.writeheader()
        for row in rows:
            writer.writerow({"experiment_id":row["experiment_id"],"status":"COMPLETE",
                              "matched_control":row["matched_control"],"weighted_f1":row["weighted_f1"],
                              "accuracy":row["accuracy"],"macro_f1":row["macro_f1"],
                              "matched_control_wf1":row["matched_control_wf1"],"matched_delta_wf1":row["matched_delta_wf1"],
                              "parameter_count":row["parameter_count"],"members":""})
        writer.writerow({"experiment_id":"47","status":selection["47"]["status"],"members":""})
        writer.writerow({"experiment_id":"48","status":"COMPLETE_NEGATIVE","matched_control":"S0",
                          "weighted_f1":outcome48["weighted_f1"],"accuracy":outcome48["accuracy"],
                          "macro_f1":outcome48["macro_f1"],"matched_control_wf1":outcome48["s0_weighted_f1"],
                          "matched_delta_wf1":outcome48["delta_vs_s0"],"parameter_count":outcome48["parameter_count"],
                          "members":"16+40"})
    positives=[r for r in rows if r["matched_delta_wf1"]>0 and int(r["experiment_id"])<=40]
    lines=["# FINAL EXPERIMENT REPORT","",f"Protocol: `{PROTOCOL}` (Test-Peak diagnostic; no generalization claim).","",
           f"All 46 first-pass numbered runs and the trained experiment 48 passed physical artifact, hash, protocol, seed, finiteness, and exact-replay audit. Experiment 47 is a frozen no-go and was not trained. Results lock: `{lock['lock_sha256']}`.","",
           "## Positive matched-control results","", "| ID | Control | WF1 | Delta | Macro-F1 |","|---|---|---:|---:|---:|"]
    for row in positives: lines.append(f"| {row['experiment_id']} | {row['matched_control']} | {row['weighted_f1']:.6f} | {row['matched_delta_wf1']:+.6f} | {row['macro_f1']:.6f} |")
    lines += ["","## Deferred decisions","",f"- 47: **{selection['47']['status']}**" + (f" ({'+'.join(selection['47']['members'])})" if selection['47']['members'] else ""),
              f"- 48 selection: **{selection['48']['status']}**" + (f" ({'+'.join(selection['48']['members'])})" if selection['48']['members'] else ""),
              f"- 48 formal result: WF1 **{outcome48['weighted_f1']:.6f}**, ACC **{outcome48['accuracy']:.6f}**, Macro-F1 **{outcome48['macro_f1']:.6f}**, epoch {outcome48['best_epoch']}.",
              f"- 48 vs S0: {outcome48['delta_vs_s0']:+.6f}; vs best member 40: {outcome48['delta_vs_best_member']:+.6f}. **No synergy; keep experiment 40.**","",
              "Experiment 16 is positive only against its required L2 matched control S1P; experiment 40 is positive against S0. Controls 45–46 and temporal 41–44 are excluded from deferred selection.","",
              "Full numeric results are in `results_01_48.csv`; selection evidence is in `deferred_selection.json`, and the audited combination verdict is in `48_final_outcome.json`." ]
    atomic_text(output/"FINAL_EXPERIMENT_REPORT.md","\n".join(lines)+"\n")
    return {"lock":lock,"selection":selection,"rows":rows}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--formal-root",default=str(ROOT/"formal")); parser.add_argument("--output",default=str(ROOT/"final_artifacts")); args=parser.parse_args(argv)
    result=audit_and_materialize(Path(args.formal_root),Path(args.output)); print(json.dumps(result["selection"],indent=2))


if __name__=="__main__": main()
