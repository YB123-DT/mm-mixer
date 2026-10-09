"""One predeclared IEMOCAP raw-concat MLP control; frozen source 1b8b1ff."""
import copy
import argparse, hashlib, json, math, os, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from source_multiattn import MultitaskFusionLoss, set_random_seed
from source_dataset import Fill53Dataset
from sklearn.utils.class_weight import compute_class_weight

ROOT=Path(__file__).resolve().parent
class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(2390,256)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(.2)
        self.classifier=nn.Linear(256,6)
    def forward(self,features):
        x=torch.cat([features[m] for m in ('t','a','v')],dim=-1)
        return self.classifier(self.dropout(self.activation(self.hidden(x))))

def loss_fn(logits,labels,weights):
    ce=torch.nn.functional.cross_entropy(logits,labels,weight=weights,reduction='none')
    pt=torch.softmax(logits,dim=-1)[range(len(labels)),labels]+1e-7
    poly=(ce+1.9508055462649292*(1-pt)*(1-pt)**1.402459950741192).mean()
    focal=(1-torch.softmax(logits,dim=-1)[range(len(labels)),labels].detach())**2.5
    return .45*(poly*focal).mean()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def metrics(y,p):
    return {'accuracy':100*accuracy_score(y,p),'weighted_f1':100*f1_score(y,p,average='weighted'), 'class_f1':(100*f1_score(y,p,labels=list(range(6)),average=None,zero_division=0)).tolist()}

@torch.no_grad()
def evaluate(model,loader):
    model.eval();ys=[];ls=[]
    for features,y in loader:
        logits=model({k:v.cuda() for k,v in features.items() if k in ('t','a','v')})
        ys.append(y.numpy());ls.append(logits.cpu().numpy())
    y=np.concatenate(ys);logits=np.concatenate(ls)
    return y,logits,metrics(y,logits.argmax(-1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists(),'No overwriting completed run'
    set_random_seed(2025);torch.set_num_threads(1)
    source=json.loads((ROOT/'source_config.json').read_text());classes=list(Fill53Dataset.LABEL_NAMES)
    config={'model':'raw_concat_mlp','dataset':'iemocap','seed':2025,'epochs':100,'batch_size':32,'gradient_accumulation_steps':2,'input_order':['t','a','v'],'input_dims':[1024,1024,342],'hidden_dim':256,'dropout':.2,'hidden_lr':6e-5,'classifier_lr':1.2e-4,'weight_decay':2e-4,'gradient_clip':1.,'scheduler':'original 20% warmup then cos(2*pi*progress), total=epochs*len(train_loader), stepped per optimizer update','selection':'strict_peak_test_wf1','auxiliary_loss':False,'ema_decay':0.999,'evaluated_model':'ema','source_commit':'1b8b1ff','training_augmentation':'original frozen Fill53Dataset defaults','main_coefficient':.45,'focal_mode':'detached batch coefficient','early_stopping':True,'original_early_stopping_patience':30,'lr_reduce_patience':5,'lr_reduce_factor':.5,'raw_feature_cache':True,'smoke_only':args.smoke}
    dump(out/'config.json',config)
    dump(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_hashes':{p.name:sha(p) for p in ROOT.glob('*') if p.is_file()}})
    pkl=Path('/data2/yb/multimodalERC/IEMOCAP/external/CSS/data/iemocap_multimodal_features.pkl')
    packed=Path('/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_textpeak_cssv_v1/artifacts/iemocap_textpeak_audio_cssv.npz')
    datasets={s:Fill53Dataset(pkl,packed,s,s=='train') for s in ('train','test')}
    assert len(datasets['train'])==5810 and len(datasets['test'])==1623
    # Cache raw arrays only; retain the unmodified stochastic __getitem__ path.
    def rng_state():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    def restore_rng(state):random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
    def same_rng(a,b):
        return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])
    original_rng=rng_state();cache_checks={}
    shared_cache={k:datasets['train'].z[k] for k in datasets['train'].z.files}
    for ds in datasets.values():
        before=rng_state();indices=[0,1,31,len(ds)//2,len(ds)-1]
        raw=ds.collate_fn([ds[i] for i in indices]);after_raw=rng_state()
        ds.z.close();ds.z=shared_cache
        restore_rng(before)
        cached=ds.collate_fn([ds[i] for i in indices]);after_cache=rng_state()
        assert torch.equal(raw[1],cached[1]) and all(torch.equal(raw[0][m],cached[0][m]) for m in ('t','a','v'))
        assert same_rng(after_raw,after_cache)
        # Returned tensors copy arrays; stochastic transforms cannot mutate cache.
        d,i=ds.index[0];saved={m:shared_cache[f'{d}__{m}'][i].copy() for m in ('t','a','v')}
        ds[0]
        assert all(np.array_equal(saved[m],shared_cache[f'{d}__{m}'][i]) for m in saved)
        cache_checks['train' if ds.is_training else 'test']={'samples_checked':indices,'batch_exact':True,'post_rng_exact':True,'cache_not_mutated':True}
    restore_rng(original_rng)
    dump(out/'cache_verification.json',cache_checks)

    manifest={'files':{str(p):sha(p) for p in (pkl,packed)}}
    for split,ds in datasets.items():
        manifest[split]={'samples':len(ds),'sample_keys':[f'{d}__{i}' for d,i in ds.index]}
    dump(out/'data_manifest.json',manifest)
    loaders={s:DataLoader(ds,batch_size=32,shuffle=s=='train',collate_fn=ds.collate_fn,num_workers=0) for s,ds in datasets.items()}
    train_labels=[datasets['train'].labels[d][i] for d,i in datasets['train'].index]
    weights=torch.tensor(compute_class_weight('balanced',classes=np.arange(6),y=np.asarray(train_labels)),device='cuda',dtype=torch.float32)
    config['class_weights']=weights.cpu().tolist();dump(out/'config.json',config)
    model=ConcatMLP().cuda();params=sum(p.numel() for p in model.parameters())
    assert params==613638,params
    # Exact value and gradient parity with frozen original main objective (empty auxiliary dict).
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    z=torch.randn(9,6,device='cuda',requires_grad=True);y=torch.arange(9,device='cuda')%6
    vendor=MultitaskFusionLoss(main_weight=.45,aux_weights={},normalize_aux_weights=False,poly_alpha=1.9508055462649292,poly_gamma=1.402459950741192,focal_gamma=2.5,ce_weight=weights,use_uncertainty=True).cuda()
    a=loss_fn(z,y,weights);b=vendor(z,{},y).reshape(())
    assert torch.equal(a,b);assert torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0])
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    optimizer=torch.optim.AdamW([{'params':model.hidden.parameters(),'lr':config['hidden_lr']},{'params':model.classifier.parameters(),'lr':config['classifier_lr']}],weight_decay=config['weight_decay'])
    total=100*len(loaders['train']);warmup=int(.2*total)
    def schedule(step):
        if step<warmup:return step/max(1,warmup)
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)*2))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    ema_model=copy.deepcopy(model)
    best=-1.;plateau=0;no_improvement=0;start=time.time();optimizer.zero_grad();initial={n:p.detach().clone() for n,p in model.named_parameters()}
    epochs=1 if args.smoke else 100
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for step,(features,y) in enumerate(loaders['train'],1):
            features={k:v.cuda() for k,v in features.items() if k in ('t','a','v')};y=y.cuda()
            logits=model(features);assert logits.shape==(len(y),6)
            loss=loss_fn(logits,y,weights);assert torch.isfinite(loss)
            (loss/2).backward();losses.append(float(loss.detach()))
            if step%2==0:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad()
                with torch.no_grad():
                    for p,ep in zip(model.parameters(),ema_model.parameters()):ep.mul_(.999).add_(p,alpha=1-.999)
            if args.smoke and step==4:break
        if step%2:optimizer.step();scheduler.step();optimizer.zero_grad()
        if epoch==1:
            updated={n:not torch.equal(initial[n],p) for n,p in model.named_parameters()};assert all(updated.values())
            dump(out/'verification.json',{'loss_value_exact':True,'loss_gradient_exact':True,'updated_parameters':updated,'parameters':params,'shape':[32,6],'finite':True})
            del initial
        yt,lt,test=evaluate(ema_model,loaders['test'])
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'test':test,'elapsed_seconds':time.time()-start}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if test['weighted_f1']>best:
            best=test['weighted_f1'];best_epoch=epoch;plateau=0;no_improvement=0
            torch.save({'model':ema_model.state_dict(),'training_model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},'config':config},out/'test_peak.pt')
            np.savez_compressed(out/'predictions.npz',labels=yt,logits=lt,predictions=lt.argmax(-1),keys=np.array(manifest['test']['sample_keys']))
        else:
            plateau+=1;no_improvement+=1
            if plateau>=5:
                plateau=0
                for group in optimizer.param_groups:group['lr']*=.5
            if no_improvement>=30:break
    epochs_completed=epoch
    fresh=ConcatMLP().cuda();saved=torch.load(out/'test_peak.pt',map_location='cuda',weights_only=False);fresh.load_state_dict(saved['model'],strict=True)
    yr,lr,m=evaluate(fresh,loaders['test']);pred=np.load(out/'predictions.npz')
    assert np.array_equal(yr,pred['labels']) and np.array_equal(lr,pred['logits'])
    result={'status':'completed','smoke_only':args.smoke,'epochs_completed':epochs_completed,'early_stopped':epochs_completed<epochs,'selected_epoch':best_epoch,'dataset':'iemocap','seed':2025,'parameters':params,'test':m,'class_names':classes,'samples':len(yr),'fresh_strict_replay_exact':True,'elapsed_seconds':time.time()-start,'checkpoint_sha256':sha(out/'test_peak.pt'),'predictions_sha256':sha(out/'predictions.npz')}
    dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
