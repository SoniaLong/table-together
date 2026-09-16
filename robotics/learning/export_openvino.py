"""Export the trained robot policy and benchmark actual inference on Intel CPU."""
from pathlib import Path
import json,time,argparse
import numpy as np
import torch
import openvino as ov
from rollout import load_policy
from train import load

ROOT=Path(__file__).resolve().parent
def export(checkpoint=None,output='checkpoints'):
    torch.set_num_threads(4)
    policy,digest=load_policy(checkpoint);val=load('validation')
    destination=ROOT/output;destination.mkdir(parents=True,exist_ok=True)
    sample=(val['images'][:1].float()/255,val['state'][:1],val['tokens'][:1],val['step'][:1])
    model=ov.convert_model(policy,example_input=sample)
    for port,name in zip(model.inputs,['image','state','tokens','step']):port.get_tensor().set_names({name})
    for port,name in zip(model.outputs,['actions','language_logits']):port.get_tensor().set_names({name})
    report={'checkpoint_sha256':digest,'openvino':ov.__version__,'device':'CPU','device_name':ov.Core().get_property('CPU','FULL_DEVICE_NAME'),'variants':{}}
    for precision,compressed in [('fp32',False),('fp16_weights',True)]:
        path=destination/f'policy-{precision}.xml'
        ov.save_model(model,path,compress_to_fp16=compressed)
        compiled=ov.Core().compile_model(str(path),'CPU',{'INFERENCE_PRECISION_HINT':'f32'})
        inputs={name:value.numpy() for name,value in zip(['image','state','tokens','step'],sample)}
        for _ in range(10):compiled(inputs)
        times=[]
        for _ in range(100):
            start=time.perf_counter();compiled(inputs);times.append((time.perf_counter()-start)*1000)
        errors=[]
        for index in range(len(val['step'])):
            args=(val['images'][index:index+1].float()/255,val['state'][index:index+1],val['tokens'][index:index+1],val['step'][index:index+1])
            with torch.no_grad():reference=policy(*args)[0].numpy()
            result=compiled({name:value.numpy() for name,value in zip(['image','state','tokens','step'],args)})['actions']
            errors.append(float(np.max(np.abs(result-reference))))
        report['variants'][precision]={'weights_bytes':path.with_suffix('.bin').stat().st_size,'warmup_calls':10,'timed_calls':100,'median_latency_ms':float(np.median(times)),'p95_latency_ms':float(np.percentile(times,95)),'serial_calls_per_second':1000/float(np.mean(times)),'maximum_action_difference_vs_pytorch':max(errors),'note':'FP16 variant compresses weights; CPU computation requested in FP32. Task-quality comparison requires rollouts.'}
    (destination/'openvino-benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--output',default='checkpoints');a=p.parse_args();export(a.checkpoint,a.output)
