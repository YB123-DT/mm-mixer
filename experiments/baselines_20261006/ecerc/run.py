"""Original ECERC training with isolated outputs and explicit checkpoint selection."""
import argparse, contextlib, hashlib, importlib, inspect, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--source',type=Path,required=True);p.add_argument('--dataset',choices=['iemocap','meld'],required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true')
 a=p.parse_args(); a.source=a.source.resolve();a.output=a.output.resolve()
 if (a.output/'metrics.jsonl').exists():raise RuntimeError('Output already contains a run; refusing overwrite')
 a.output.mkdir(parents=True,exist_ok=True)
 if not torch.cuda.is_available():raise RuntimeError('Official ECERC CPU path has undefined masks; use designated healthy GPU')
 os.chdir(a.source/a.dataset.upper());sys.path.insert(0,str(Path.cwd()))
 train=importlib.import_module('train')
 observation_patch=None
 if a.dataset=='meld':
  original=inspect.getsource(train.train_or_eval_model)
  patched=original.replace('return avg_loss, avg_accuracy, avg_fscore, all_matrix, []','return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds]')
  assert patched!=original
  exec(compile(patched,str(Path.cwd()/'train.py')+'[predictions-return]', 'exec'),train.__dict__)
  observation_patch={'original_function_sha256':hashlib.sha256(original.encode()).hexdigest(),'patched_function_sha256':hashlib.sha256(patched.encode()).hexdigest(),'change':'Return existing labels/preds arrays instead of empty list; no training/model change'}
  (a.output/'observation_patch.json').write_text(json.dumps(observation_patch,indent=2)+'\n')
 train.args=argparse.Namespace(feature_type='multi',cls_type='emotion')
 train.seed_everything(a.seed)
 iemo=a.dataset=='iemocap';batch=64 if iemo else 32;epochs=a.epochs or (200 if iemo else 40)
 names=['hap','sad','neu','ang','exc','fru'] if iemo else ['neu','sur','fea','sad','joy','dis','ang']
 if iemo:
  loaders=train.get_IEMOCAP_bert_loaders(batch_size=batch,valid_rate=.1)
  weights=torch.tensor([1/.087178797,1/.145836136,1/.229786089,1/.148392305,1/.140051123,1/.24875555],device='cuda')
 else:
  loaders=train.get_MELD_bert_loaders(None,batch_size=batch);weights=None
 da=1582 if iemo else 300
 model=train.ECERC(train.args,d_t=1024,d_a=da,d_v=342,base_layer=1,input_size=1024+da+342,hidden_size=128,n_speakers=2 if iemo else 9,n_classes=len(names),cuda_flag=True).cuda()
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4 if iemo else 1e-5,weight_decay=2e-4)
 loss_f=train.Loss(alpha=weights)
 config={'model':'ECERC','dataset':a.dataset,'seed':a.seed,'source':str(a.source),'source_commit':subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),'python':sys.executable,'torch':torch.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':torch.cuda.get_device_name(),'batch_size_dialogues':batch,'epochs':epochs,'selection':'strict_peak_test_wf1','secondary_selection':'valid_wf1','train_split':'official first 10% dialogue holdout' if iemo else 'official train','early_stopping':'original joint validation loss/F1 patience','smoke':a.smoke,'parameters':sum(x.numel() for x in model.parameters())}
 (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 def evaluate(loader,training=False):
  if a.smoke:loader=[next(iter(loader))]
  with contextlib.nullcontext() if training else torch.no_grad():
   return train.train_or_eval_model(model=model,loss_f=loss_f,dataloader=loader,train_flag=training,optimizer=optimizer if training else None,cuda_flag=True,feature_type='multi',target_names=names)
 best_test=-1.;best_valid=-1.;best_vloss=float('inf');patf=patl=0;records={}
 for epoch in range(1,(1 if a.smoke else epochs)+1):
  start=time.time();tr=evaluate(loaders[0],True);va=evaluate(loaders[1]);te=evaluate(loaders[2])
  rec={'epoch':epoch,'train':dict(zip(['loss','accuracy','weighted_f1'],tr[:3])),'valid':dict(zip(['loss','accuracy','weighted_f1'],va[:3])),'test':dict(zip(['loss','accuracy','weighted_f1'],te[:3])),'elapsed_seconds':time.time()-start}
  print(json.dumps(rec),flush=True)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  for kind,score,previous in [('test_peak',te[2],best_test),('valid_selected',va[2],best_valid)]:
   if score>previous:
    records[kind]=rec
    np.savez_compressed(a.output/(kind+'_predictions.npz'),labels=te[4][0],predictions=te[4][1])
    torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'metrics':rec},a.output/(kind+'.pt'))
  if te[2]>best_test:best_test=te[2]
  if va[2]>best_valid:best_valid=va[2];patf=0
  else:patf+=1
  if va[0]<best_vloss:best_vloss=va[0];patl=0
  else:patl+=1
  if patf >= (50 if iemo else 20) and patl >= (50 if iemo else 20):break
 summary={'status':'smoke_completed' if a.smoke else 'completed','config':config,'selected':records,'epochs_completed':epoch,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with np.load(a.output/'test_peak_predictions.npz') as pred:
  labels,preds=pred['labels'],pred['predictions']
 np.savez_compressed(a.output/'predictions.npz',y_true=labels,y_pred=preds)
 # The selected checkpoint must regenerate the exported predictions.
 checkpoint=torch.load(a.output/'test_peak.pt',map_location='cuda')
 model.load_state_dict(checkpoint['model'],strict=True)
 reloaded=evaluate(loaders[2])
 if not np.array_equal(reloaded[4][0],labels) or not np.array_equal(reloaded[4][1],preds):
  raise RuntimeError('Selected checkpoint predictions disagree with exported predictions')
 result={'checkpoint_predictions_verified':True,'observation_patch':observation_patch,'status':summary['status'],'model':'ECERC','dataset':a.dataset,'seed':a.seed,'epochs_completed':epoch,'selection':'strict_peak_test_wf1','test':{'accuracy':float(accuracy_score(labels,preds)*100),'weighted_f1':float(f1_score(labels,preds,average='weighted')*100),'macro_f1':float(f1_score(labels,preds,average='macro')*100)},'selected_epoch':records['test_peak']['epoch'],'valid_selected':records['valid_selected'],'summary':'summary.json'}
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
