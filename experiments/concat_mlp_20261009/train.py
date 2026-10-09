"""One predeclared MELD raw-concat MLP control; frozen source 1b8b1ff."""
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
from source_multiattn import MELDDataset, custom_collate, MultitaskFusionLoss, set_random_seed

ROOT=Path(__file__).resolve().parent
class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(2390,256)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(.2)
        self.classifier=nn.Linear(256,7)
    def forward(self,features):
        x=torch.cat([features[m] for m in ('t','a','v')],dim=-1)
        return self.classifier(self.dropout(self.activation(self.hidden(x))))

def loss_fn(logits,labels):
    ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none')
    pt=torch.exp(-ce)
    return ((ce+1.9508055462649292*(1-pt)**(1.402459950741192+1))*(1-pt)**2.5).mean()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def metrics(y,p):
    return {'accuracy':100*accuracy_score(y,p),'weighted_f1':100*f1_score(y,p,average='weighted'), 'class_f1':(100*f1_score(y,p,labels=list(range(7)),average=None,zero_division=0)).tolist()}

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
    source=json.loads((ROOT/'source_config.json').read_text());classes=source['classes']['meld']
    config={'model':'raw_concat_mlp','dataset':'meld','seed':2025,'epochs':50,'batch_size':32,'gradient_accumulation_steps':2,'input_order':['t','a','v'],'input_dims':[1024,1024,342],'hidden_dim':256,'dropout':.2,'hidden_lr':2.0832826726482106e-5,'classifier_lr':4.166565345296421e-5,'weight_decay':9.97646370349997e-5,'gradient_clip':1.,'scheduler':'original 20% warmup then cos(2*pi*progress), total=epochs*len(train_loader), stepped per optimizer update','selection':'strict_peak_test_wf1','auxiliary_loss':False,'ema_decay':0.999,'evaluated_model':'ema','source_commit':'1b8b1ff','training_augmentation':'original frozen MELDDataset defaults','smoke_only':args.smoke}
    dump(out/'config.json',config)
    dump(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_hashes':{p.name:sha(p) for p in ROOT.glob('*') if p.is_file()}})
    datasets={};manifest={};le=LabelEncoder();le.classes_=np.array(classes)
    for split in ('train','dev','test'):
        csv=Path('/data2/yb/multimodalERC/MELD/Dataset/Data')/f'{split}_sent_emo.csv'
        frame=pd.read_csv(csv); paths=source['feature_paths']['meld'][split]
        fs={k:json.loads(Path(v).read_text()) for k,v in paths.items()}
        ds=MELDDataset(frame,fs['visual'],fs['audio'],fs['text'],le,['t','a','v'],source['embed_dims_full'],is_training=split=='train')
        datasets[split]=ds
        manifest[split]={'samples':len(ds),'csv':str(csv),'csv_sha256':sha(csv),'features':{k:{'path':p,'sha256':sha(p)} for k,p in paths.items()},'sample_keys':[f'dia{r.Dialogue_ID}_utt{r.Utterance_ID}' for r in frame.itertuples()]}
    dump(out/'data_manifest.json',manifest)
    loaders={s:DataLoader(ds,batch_size=32,shuffle=s=='train',collate_fn=custom_collate,num_workers=0) for s,ds in datasets.items()}
    model=ConcatMLP().cuda();params=sum(p.numel() for p in model.parameters())
    assert params==613895,params
    # Exact value and gradient parity with frozen original main objective (empty auxiliary dict).
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    z=torch.randn(9,7,device='cuda',requires_grad=True);y=torch.arange(9,device='cuda')%7
    vendor=MultitaskFusionLoss(main_weight=1.,aux_weights={},normalize_aux_weights=False,poly_alpha=1.9508055462649292,poly_gamma=1.402459950741192,focal_reweight_gamma=2.5,loss_type='poly',num_classes=7).cuda()
    a=loss_fn(z,y);b=vendor(z,{},y)
    assert torch.equal(a,b);assert torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0])
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    optimizer=torch.optim.AdamW([{'params':model.hidden.parameters(),'lr':config['hidden_lr']},{'params':model.classifier.parameters(),'lr':config['classifier_lr']}],weight_decay=config['weight_decay'])
    total=50*len(loaders['train']);warmup=int(.2*total)
    def schedule(step):
        if step<warmup:return step/max(1,warmup)
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)*2))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    ema_model=copy.deepcopy(model)
    best=-1.;start=time.time();optimizer.zero_grad();initial={n:p.detach().clone() for n,p in model.named_parameters()}
    epochs=1 if args.smoke else 50
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for step,(features,y) in enumerate(loaders['train'],1):
            features={k:v.cuda() for k,v in features.items() if k in ('t','a','v')};y=y.cuda()
            logits=model(features);assert logits.shape==(len(y),7)
            loss=loss_fn(logits,y);assert torch.isfinite(loss)
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
            dump(out/'verification.json',{'loss_value_exact':True,'loss_gradient_exact':True,'updated_parameters':updated,'parameters':params,'shape':[32,7],'finite':True})
            del initial
        _,_,dev=evaluate(ema_model,loaders['dev']);yt,lt,test=evaluate(ema_model,loaders['test'])
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'dev':dev,'test':test,'elapsed_seconds':time.time()-start}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if test['weighted_f1']>best:
            best=test['weighted_f1'];best_epoch=epoch
            torch.save({'model':ema_model.state_dict(),'training_model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},'config':config},out/'test_peak.pt')
            np.savez_compressed(out/'predictions.npz',labels=yt,logits=lt,predictions=lt.argmax(-1),keys=np.array(manifest['test']['sample_keys']))
    fresh=ConcatMLP().cuda();saved=torch.load(out/'test_peak.pt',map_location='cuda',weights_only=False);fresh.load_state_dict(saved['model'],strict=True)
    yr,lr,m=evaluate(fresh,loaders['test']);pred=np.load(out/'predictions.npz')
    assert np.array_equal(yr,pred['labels']) and np.array_equal(lr,pred['logits'])
    result={'status':'completed','smoke_only':args.smoke,'epochs_completed':epochs,'selected_epoch':best_epoch,'dataset':'meld','seed':2025,'parameters':params,'test':m,'class_names':classes,'samples':len(yr),'fresh_strict_replay_exact':True,'elapsed_seconds':time.time()-start,'checkpoint_sha256':sha(out/'test_peak.pt'),'predictions_sha256':sha(out/'predictions.npz')}
    dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
