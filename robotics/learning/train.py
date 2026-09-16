"""CPU training with episode-level train/validation separation and saved provenance."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from policy import Policy,tokenize,INSTRUCTIONS
ROOT=Path(__file__).resolve().parent

def load(split):
    files=sorted((ROOT/'data'/split).glob('*.npz'))
    if not files:raise RuntimeError(f'No successful demonstrations in {split}')
    episodes=[dict(np.load(path)) for path in files]
    return {'images':torch.from_numpy(np.concatenate([e['images'] for e in episodes])).permute(0,3,1,2),
            'state':torch.from_numpy(np.concatenate([e['state'] for e in episodes])),
            'actions':torch.from_numpy(np.concatenate([e['actions'] for e in episodes])),
            'tokens':torch.from_numpy(np.concatenate([np.tile(tokenize(INSTRUCTIONS[int(e['task'])]),(18,1)) for e in episodes])),
            'step':torch.arange(18).repeat(len(episodes)),
            'task':torch.tensor([int(e['task']) for e in episodes]).repeat_interleave(18),
            'files':[str(f.relative_to(ROOT)) for f in files],
            'seeds':sorted(set(int(e['seed']) for e in episodes))}

def train(iterations=1200,output='checkpoints',version=1,warmstart=None):
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    trainset=load('train');val=load('validation')
    assert not set(trainset['seeds'])&set(val['seeds'])
    actions=trainset['actions'].reshape(-1,18,13)
    labels=trainset['task'][::18]
    means=torch.stack([actions[labels==i].mean(0) for i in range(2)])
    std=torch.stack([actions[labels==i].std(0) for i in range(2)])
    scales=torch.where(std>1e-5,std*3,torch.zeros_like(std))
    net=Policy(means,scales,version)
    if warmstart:
        saved=torch.load(warmstart,map_location='cpu',weights_only=True)['state_dict']
        current=net.state_dict()
        for key,value in saved.items():
            if key in ('means','scales'):continue
            if value.shape==current[key].shape:current[key]=value
            elif key=='head.4.weight' and version==2:current[key]=value.repeat(18,1)
            elif key=='head.4.bias' and version==2:current[key]=value.repeat(18)
        net.load_state_dict(current)
    optimizer=torch.optim.Adam(net.parameters(),lr=.0002 if warmstart else .001)
    out=ROOT/output;out.mkdir(exist_ok=True)
    started=time.perf_counter();history=[];best=float('inf')
    # Teach the token encoder the two requested orders before action regression.
    for _ in range(100):
        language=net.language(net.words(trainset['tokens'][::18]))
        loss=torch.nn.functional.cross_entropy(net.task(language),labels)
        optimizer.zero_grad();loss.backward();optimizer.step()
    normalization=std.mean((0,1)).clamp_min(.005)
    for iteration in range(iterations):
        idx=torch.randint(len(trainset['step']),(min(64,len(trainset['step'])),))
        images=trainset['images'][idx].float()/255
        state=trainset['state'][idx]
        if version==2:
            images=(images*(.85+.3*torch.rand(len(idx),1,1,1))).clamp(0,1)
            state=state+.001*torch.randn_like(state)
        pred,logits=net(images,state,trainset['tokens'][idx],trainset['step'][idx])
        loss=(((pred-trainset['actions'][idx])/normalization)**2).mean()+.1*torch.nn.functional.cross_entropy(logits,trainset['task'][idx])
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
        if iteration%100==0 or iteration==iterations-1:
            net.eval()
            with torch.no_grad():
                predictions=[]
                for lo in range(0,len(val['step']),64):
                    sl=slice(lo,lo+64)
                    predictions.append(net(val['images'][sl].float()/255,val['state'][sl],val['tokens'][sl],val['step'][sl])[0])
                predicted=torch.cat(predictions)
                mse=float((((predicted-val['actions'])/normalization)**2).mean())
                mae=float((predicted[:,:12]-val['actions'][:,:12]).abs().mean())
            row={'iteration':iteration,'train_loss':float(loss.detach()),'validation_normalized_mse':mse,'validation_joint_mae_rad':mae,'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
            if mse<best:
                best=mse
                torch.save({'state_dict':net.state_dict(),'means':means,'scales':scales,'version':version,'iteration':iteration,'validation_mse':mse},out/'policy.pt')
            (out/'training.json').write_text(json.dumps({'architecture':'RGB + token encoder + proprioception + ordinal history -> joint keyframe; learned demonstration means and residuals','torch':torch.__version__,'device':'CPU','train_files':trainset['files'],'validation_files':val['files'],'train_seeds':trainset['seeds'],'validation_seeds':val['seeds'],'history':history},indent=2))
            net.train()
    checkpoint=out/'policy.pt'
    (out/'SHA256').write_text(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    print('Saved '+str(checkpoint),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=1200);p.add_argument('--output',default='checkpoints');p.add_argument('--version',type=int,choices=[1,2],default=1);p.add_argument('--warmstart');a=p.parse_args();train(a.iterations,a.output,a.version,a.warmstart)
