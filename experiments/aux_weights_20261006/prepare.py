#!/usr/bin/env python3
import subprocess,io,tarfile,difflib,json,hashlib
from pathlib import Path
BASE='1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab'
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO/'outputs/aux_weights_20261006/code'
ROOT.mkdir(parents=True,exist_ok=True)
raw=subprocess.check_output(['git','-C',str(REPO),'archive',BASE])
with tarfile.open(fileobj=io.BytesIO(raw)) as t:t.extractall(ROOT)
original={str(p.relative_to(ROOT)):p.read_text() for p in ROOT.rglob('*.py')}
def edit(name,old,new):
 p=ROOT/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new))
(ROOT/'mm_mixer_final/aux_weights.py').write_text('''"""Prespecified auxiliary-weight sensitivity; no normalization."""
import copy
AUX_WEIGHT_VARIANTS = ("aux_half", "aux_double", "aux_equal")
def weights(dataset, variant):
    value = {"main": .45, "t": .33, "a": .11, "v": .11} if dataset == "iemocap" else {"main": 1., "t": 1., "a": 1., "v": 1.}
    if variant == "aux_equal":
        if dataset != "iemocap": raise ValueError("MELD is already equal weighted")
        value.update({m: .55 / 3 for m in ("t", "a", "v")})
    elif variant in ("aux_half", "aux_double"):
        scale = .5 if variant == "aux_half" else 2.
        value.update({m: value[m] * scale for m in ("t", "a", "v")})
    return value

def apply_weights(fixed, dataset, variant):
    if variant not in AUX_WEIGHT_VARIANTS: return fixed
    fixed = copy.deepcopy(fixed)
    w = weights(dataset, variant)
    fixed.update(main_loss_weight=w["main"], aux_loss_weights={m:w[m] for m in ("t","a","v")}, normalize_aux_loss_weights=False)
    return fixed
''')
edit('mm_mixer_final/config.py','from .modalities import','from .aux_weights import AUX_WEIGHT_VARIANTS, weights\nfrom .modalities import')
edit('mm_mixer_final/config.py','RUN_VARIANTS = VARIANTS + MODALITY_VARIANTS + STRUCTURAL_ABLATION_VARIANTS + REVISION_CONTROL_VARIANTS','RUN_VARIANTS = VARIANTS + MODALITY_VARIANTS + STRUCTURAL_ABLATION_VARIANTS + REVISION_CONTROL_VARIANTS + AUX_WEIGHT_VARIANTS')
edit('mm_mixer_final/config.py','return replace(base, variant=variant, seed=int(seed), switches=switches, mixer=mixer)','loss = dict(base.loss)\n    if variant in AUX_WEIGHT_VARIANTS:\n        loss["weights" if dataset == "iemocap" else "fixed_task_weights"] = weights(dataset, variant)\n        loss["task_weighting"] = "fixed_prespecified_aux_sensitivity"\n    return replace(base, variant=variant, seed=int(seed), switches=switches, mixer=mixer, loss=loss)')
for d in ('iemocap','meld'):
 edit('dataset_runners/'+d+'.py','from mm_mixer_final.config import','from mm_mixer_final.aux_weights import AUX_WEIGHT_VARIANTS, apply_weights\nfrom mm_mixer_final.config import')
edit('dataset_runners/iemocap.py','        *REVISION_CONTROL_VARIANTS,','        *REVISION_CONTROL_VARIANTS,\n        *AUX_WEIGHT_VARIANTS,')
edit('dataset_runners/iemocap.py','    return fixed\n','    return apply_weights(fixed, "iemocap", variant)\n')
edit('dataset_runners/meld.py','    **{variant: "M4_PAIR" for variant in REVISION_CONTROL_VARIANTS},','    **{variant: "M4_PAIR" for variant in REVISION_CONTROL_VARIANTS},\n    **{variant: "M4_PAIR" for variant in AUX_WEIGHT_VARIANTS},')
edit('dataset_runners/meld.py','    config["runtime_audit"] = {','    fixed.update(apply_weights(fixed, "meld", variant))\n    config["runtime_audit"] = {')
# Historical MELD code ignores the configured numeric coefficients. Preserve unit path exactly.
edit('vendor/meld/multiattn.py','        total_loss = weighted_main_loss\n        for m, logits in aux_logits.items():\n            if m in self.aux_weights:\n                total_loss += self.ce_loss(logits, targets)','        total_loss = weighted_main_loss if self.main_weight == 1.0 else self.main_weight * weighted_main_loss\n        for m, logits in aux_logits.items():\n            if m in self.aux_weights:\n                aux_loss = self.ce_loss(logits, targets)\n                total_loss += aux_loss if self.aux_weights[m] == 1.0 else self.aux_weights[m] * aux_loss')
patch=''
for p in sorted(ROOT.rglob('*.py')):
 name=str(p.relative_to(ROOT));before=original.get(name,'');after=p.read_text()
 if before!=after:patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name if name in original else '/dev/null',tofile='b/'+name))
(HERE/'aux_weights.patch').write_text(patch)
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='snapshot.json'}
(ROOT/'snapshot.json').write_text(json.dumps({'base_commit':BASE,'file_sha256':hashes,'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()},indent=2)+'\n')
