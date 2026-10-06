"""Checkpoint loaders copied narrowly from verified FLOPs adapters; no model edits."""
import ast
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

OLD = Path('/data2/yb/paper/tsne_baselines_20260728')
NEW = Path('/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006')


def enter(repo):
    sys.path.insert(0, str(repo))
    os.chdir(repo)


def load(name, dataset, device):
    """Return model, unpadded dialog records, collate, provenance, forward adapter."""
    iemo = dataset == 'iemocap'
    batch = 16 if name == 'confilmer' else 32
    cp = OLD/'results'/name/dataset/'model.pt'
    mode = 'trained_checkpoint'
    cfg = {}
    if name == 'dialoguernn':
        repo = OLD/'DialogueRNN'; enter(repo)
        train = importlib.import_module('benchmark_checkpoint')
        dl = importlib.import_module('dataloader')
        saved = torch.load(cp, map_location='cpu')
        model = train.make_model(dataset, saved['causal_history'])
        model.load_state_dict(saved['state_dict'], strict=True)
        label = dataset.upper()
        feature = repo/f'DialogueRNN_features/{label}_features/{label}_features_raw.pkl'
        ds = dl.IEMOCAPDataset(path=str(feature), train=False) if iemo else dl.MELDDataset(path=str(feature), n_classes=7, train=False)
        cfg = {'causal_history': saved['causal_history'], 'modalities': 'T' if iemo else 'TA'}
        def forward(data):
            if iemo: t,v,a,q,u,y = data[:-1]
            else: t,a,q,u,y = data[:-1]
            return model(t if iemo else torch.cat((t,a),-1),q,u,att2=True)
    elif name == 'ada2i':
        repo = OLD/'Ada2I'; enter(repo)
        dl = importlib.import_module('dataloader')
        cp = repo/f'checkpoint/{dataset}_causal_s13_best.pt'
        saved = torch.load(cp,map_location='cpu'); args = saved['args']
        args.device=device;args.dataset=dataset;args.batch_size=batch;args.causal_history=True
        cfg=vars(args)
        model=importlib.import_module('model').Ada2I(args)
        model.load_state_dict(saved['state_dict'],strict=True)
        raw=(dl.load_iemocap() if iemo else dl.load_meld())['test']
        collate=dl.Dataloader(raw,args).padding
        def forward(data): return model(data)
        return model,raw,collate,dict(repo=str(repo),checkpoint=str(cp),weight_source=mode,configuration=cfg,batch_size_dialogues=batch,raw_format='ada2i'),forward
    elif name in ('mmgcn','mmdfn','m3net'):
        repo=OLD/{'mmgcn':'MMGCN','mmdfn':'MM-DFN/code','m3net':'M3NET'}[name]; enter(repo)
        saved=torch.load(cp,map_location='cpu')
        metrics=json.loads((cp.parent/'metrics.json').read_text());cfg=metrics
        if name=='mmgcn':
            model=importlib.import_module('export_checkpoint').build_model(SimpleNamespace(**saved['args']))
            model.load_state_dict(saved['state_dict'],strict=True)
            dl=importlib.import_module('dataloader')
            ds=dl.IEMOCAPDataset(train=False) if iemo else dl.MELDDataset('MELD_features/MELD_features_raw1.pkl',train=False)
        else:
            model=saved['model'];train=importlib.import_module('run_train_erc' if name=='mmdfn' else 'train')
            opts=dict(batch_size=batch,num_workers=0)
            if name=='mmdfn':opts.update(data_path=metrics['command_args']['data_dir'],valid_rate=0.)
            else:opts['valid']=0.
            loader=getattr(train,'get_'+dataset.upper()+'_loaders')(**opts)[2];ds=loader.dataset
        for mod in model.modules():
            if hasattr(mod,'no_cuda'):mod.no_cuda=device=='cpu'
        def forward(data):
            lengths=data.lengths
            if name=='m3net':
                t1,t2,t3,t4,v,a,q,u,y=data[:-1]
                return model([t1,t2,t3,t4],q,u,lengths,a,v,int(saved['epoch']))
            t,v,a,q,u,y=data[:-1]
            return model(t,q,u,lengths,a,v,*([False] if name=='mmdfn' else []))
    elif name in ('sdt','css'):
        if name=='sdt':
            repo=OLD/'SDT';cfg=dict(temp=1 if iemo else 8,hidden_dim=1024,n_head=8,dropout=.5,causal_context=True)
        else:
            run=Path('/data2/yb/reproduction_workspace/runs/CSS/causal_iemocap_10seeds/seed61080') if iemo else Path('/data2/yb/reproduction_workspace/runs/paper_meld_seed_queue_20260724/css_meld_causal/seed10073')
            repo=run/'code' if iemo else Path('/data2/yb/paper/CSS_causal_meld')
            node=ast.parse((run/'train.log').read_text().splitlines()[0],mode='eval').body
            logged={k.arg:ast.literal_eval(k.value) for k in node.keywords}
            cfg={k:logged[k] for k in ['temp','hidden_dim','n_head','dropout','causal_context','rank','order']}
            cp=Path(json.loads((OLD/'results/css'/dataset/'metrics.json').read_text())['checkpoint'])
        enter(repo);dl=importlib.import_module('dataloader')
        ds=dl.IEMOCAPDataset(train=False) if iemo else dl.MELDDataset('data/meld_multimodal_features.pkl',train=False)
        args=[dataset.upper(),cfg['temp'],1024,342,1582 if iemo else 300,cfg['n_head']]
        if name=='css':args.extend([cfg['rank'],cfg['order']])
        model=importlib.import_module('model').Transformer_Based_Model(*args,n_classes=6 if iemo else 7,hidden_dim=cfg['hidden_dim'],n_speakers=2 if iemo else 9,dropout=cfg['dropout'],causal_context=True)
        if cp.is_file():
            saved=torch.load(cp,map_location='cpu');model.load_state_dict(saved['model_state_dict'] if name=='css' else saved,strict=True)
        else:
            assert name=='css';mode='architecture_reconstructed_missing_checkpoint'
            assert sum(p.numel() for p in model.parameters())==(50452563 if iemo else 53398634)
        def forward(data):
            t,v,a,q,u,y=data[:-1]
            return model(t,v,a,u,q.permute(1,0,2),data.lengths)
    elif name in ('ecerc','confilmer'):
        plan=json.loads((NEW/'plans/recovery_12_runs.json').read_text())
        job=next(j for j in plan['jobs'] if j['id']==f'{name}_{dataset}_seed2025')
        cp=Path(job['output'])/('test_peak.pt' if name=='ecerc' else 'best.pt')
        saved=torch.load(cp,map_location='cpu',weights_only=False);cfg=saved['config'];args=SimpleNamespace(**cfg)
        repo=NEW/('code/ECERC/'+dataset.upper() if name=='ecerc' else 'source/confilmer');enter(repo)
        train=importlib.import_module('train' if name=='ecerc' else 'train_our')
        if name=='ecerc':
            model=train.ECERC(SimpleNamespace(feature_type='multi',cls_type='emotion'),d_t=1024,d_a=1582 if iemo else 300,d_v=342,base_layer=1,input_size=1024+(1582 if iemo else 300)+342,hidden_size=128,n_speakers=2 if iemo else 9,n_classes=6 if iemo else 7,cuda_flag=True)
            loader=train.get_IEMOCAP_bert_loaders(batch_size=batch,valid_rate=.1)[2] if iemo else train.get_MELD_bert_loaders(None,batch_size=batch)[2]
            def forward(data):
                e,s,a,v,q,u,y=data[:-1]
                return model(torch.cat([e,a,v],-1),s,q,u,data.lengths)
        else:
            model=train.Model(args.base_model,1024,512 if iemo else 1024,150,100,100,100,512,n_speakers=2 if iemo else 9,max_seq_len=200,window_past=args.windowp,window_future=args.windowf,n_classes=6 if iemo else 7,listener_state=args.active_listener,context_attention=args.attention,dropout=args.dropout,nodal_attention=args.nodal_attention,no_cuda=device=='cpu',graph_type=args.graph_type,use_topic=args.use_topic,alpha=args.alpha,multiheads=args.multiheads,graph_construct=args.graph_construct,use_GCN=args.use_gcn,use_residue=args.use_residue,D_m_v=342,D_m_a=1582 if iemo else 300,modals=args.modals,att_type=args.mm_fusion_mthd,av_using_lstm=args.av_using_lstm,Deep_GCN_nlayers=args.Deep_GCN_nlayers,dataset=args.Dataset,use_speaker=args.use_speaker,use_modal=args.use_modal,norm=args.norm,num_L=args.num_L,num_K=args.num_K)
            loader=getattr(train,'get_'+dataset.upper()+'_loaders')(batch_size=batch,valid=0.,num_workers=0)[2]
            def forward(data):
                t1,t2,t3,t4,v,a,q,u,y,sentence=data[:-1]
                return model([t1,t2,t3,t4],q,u,data.lengths,sentence,torch.nn.Identity(),a,v,saved['epoch']-1)
        ds=loader.dataset;model.load_state_dict(saved['model'],strict=True)
    else:raise ValueError(name)
    raw=[ds[i] for i in range(len(ds))]
    return model,raw,ds.collate_fn,dict(repo=str(repo),checkpoint=str(cp),weight_source=mode,configuration=cfg,batch_size_dialogues=batch,raw_format='tuple'),forward
