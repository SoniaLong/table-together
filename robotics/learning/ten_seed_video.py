"""Record ten consecutive held-out seeds, retaining failures and traceable reports."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageSequence
import imageio_ffmpeg
from rollout import rollout,load_policy
from openvino_policy import OpenVINOPolicy

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--runtime',choices=['torch','openvino'],default='openvino');p.add_argument('--start',type=int,default=600);p.add_argument('--label',default='ten-seed-final');p.add_argument('--reuse',action='store_true');a=p.parse_args()
torch.set_num_threads(4)
if a.runtime=='openvino':policy=OpenVINOPolicy();digest=policy.digest
else:policy,digest=load_policy(a.checkpoint)
out=ROOT/'evaluations'/a.label;out.mkdir(parents=True,exist_ok=True)
video=out/'TableTogether-ten-seeds.mp4'
if a.reuse:
    saved=json.loads((out/'summary.json').read_text())
    assert saved['checkpoint_sha256']==digest and saved['episodes']==10
font=ImageFont.load_default(size=17)
writer=imageio_ffmpeg.write_frames(str(video),(640,608),fps=10,codec='libx264',pix_fmt_out='yuv420p',output_params=['-movflags','+faststart'])
writer.send(None)
results=[]
try:
    for seed in range(a.start,a.start+10):
        directory=out/str(seed)
        result=json.loads((directory/'rollout.json').read_text()) if a.reuse else rollout(policy,seed,task=0,video=True,out=directory,video_size=(320,240))
        results.append(result)
        with Image.open(directory/'rollout.gif') as movie:
            frames=[f.convert('RGB').copy() for f in ImageSequence.Iterator(movie)]
        for index,frame in enumerate(frames+[frames[-1]]*20):
            canvas=Image.new('RGB',(640,608),'#10171c');canvas.paste(frame.resize((640,480)),(0,110))
            draw=ImageDraw.Draw(canvas)
            draw.text((16,10),f'TableTogether | Seed {seed} | {a.runtime} CPU | ~2x',fill='white',font=font)
            draw.text((16,34),'Command: Set the table. Start with the cup.',fill='white',font=font)
            draw.text((16,58),'XY +/-8 mm | Frozen model '+digest[:12],fill='white',font=font)
            final=index>=len(frames)
            draw.text((16,82),('FINAL: '+('PASS' if result['success'] else 'FAIL')) if final else 'Actual contact simulation; no expert fallback',fill='#75e0b8' if not final or result['success'] else '#ff8585',font=font)
            writer.send(np.asarray(canvas).tobytes())
        (out/'summary.json').write_text(json.dumps({'checkpoint_sha256':digest,'runtime':a.runtime,'seeds':list(range(a.start,a.start+10)),'successes':sum(x['success'] for x in results),'episodes':len(results),'results':results},indent=2))
        print(json.dumps({'seed':seed,'success':result['success'],'completed':len(results)}),flush=True)
    for _ in range(50):
        canvas=Image.new('RGB',(640,608),'#10171c');draw=ImageDraw.Draw(canvas)
        draw.text((30,40),f'Ten-seed result: {sum(x["success"] for x in results)}/10',fill='white',font=font)
        for i,r in enumerate(results):draw.text((30,80+i*30),f'Seed {r["seed"]}: '+('PASS' if r['success'] else 'FAIL'),fill='#75e0b8' if r['success'] else '#ff8585',font=font)
        draw.text((30,430),'Local Series 1 run; Series 2/3 not verified.',fill='white',font=font)
        draw.text((30,460),'Position randomization only. All ten outcomes included.',fill='white',font=font)
        writer.send(np.asarray(canvas).tobytes())
finally:writer.close()
print(str(video),flush=True)
