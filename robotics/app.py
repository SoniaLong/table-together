"""Local-only demo UI. One physical simulation job at a time."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import threading,json,os,sys
from scene import ROOT

state={'running':False,'stage':'Ready','result':None}
lock=threading.Lock()
def job(instruction,seed,mode):
    try:
        def event(e):
            with lock:state['stage']=e['object']+' / '+e['stage']
        if mode=='learned':
            sys.path.insert(0,str(ROOT/'learning')) if str(ROOT/'learning') not in sys.path else None
            from openvino_policy import OpenVINOPolicy
            from rollout import rollout
            from policy import INSTRUCTIONS
            import torch
            torch.set_num_threads(4)
            normalized=instruction.strip().lower()
            supported=[s.lower() for s in INSTRUCTIONS]
            if normalized not in supported:raise ValueError('Learned mode currently supports: '+ ' OR '.join(INSTRUCTIONS))
            task=supported.index(normalized)
            with lock:state['stage']='Loading trained policy through OpenVINO'
            policy=OpenVINOPolicy(os.environ.get('ROBOT_DEVICE','CPU'))
            with lock:state['plan']={'model':'TableTogether trained vision-language keyframe policy','runtime':'OpenVINO','device':policy.device,'checkpoint_sha256':policy.digest,'instruction':instruction}
            learned=rollout(policy,seed=seed,task=task,video=True,out=ROOT/'artifacts',on_event=event,instruction=instruction)
            result=[{'evaluation':name,'placed':value['passed'],**value} for name,value in learned.get('objects',{}).items()]
            result.append({key:value for key,value in learned.items() if key not in ['actions','objects']})
        else:
            from planner import Planner
            from episode import run
            with lock:state['stage']='Loading baseline language model'
            plan=Planner(os.environ.get('ROBOT_DEVICE','CPU')).plan(instruction)
            with lock:state['plan']=plan
            result=run(event,seed=seed)
        with lock:state.update(stage='Place setting complete' if result[-1]['success'] else 'Incomplete — inspect run evidence',result=result)
    except Exception as exc:
        with lock:state.update(stage='Run failed',error=str(exc))
    finally:
        with lock:
            state['running']=False
            (ROOT/'artifacts/full-run.json').write_text(json.dumps(state,indent=2))

class Handler(BaseHTTPRequestHandler):
    def respond(self,payload,kind='application/json',code=200):
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/':return self.respond((ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/status':
            with lock:payload=json.dumps(state).encode()
            return self.respond(payload)
        if path=='/report':
            report=ROOT/'artifacts/full-run.json'
            if report.exists():return self.respond(report.read_bytes())
            return self.respond(b'{"error":"No completed run yet"}',code=404)
        if path=='/frame':
            frame=ROOT/'artifacts/live.jpg'
            if frame.exists():return self.respond(frame.read_bytes(),'image/jpeg')
            frame=ROOT/'artifacts/overview.png'
            if frame.exists():return self.respond(frame.read_bytes(),'image/png')
        return self.respond(b'{}',code=404)
    def do_POST(self):
        if self.path!='/run':return self.respond(b'{}',code=404)
        # JSON-only API plus same-origin check avoids cross-site form-triggered runs.
        if self.headers.get('Content-Type')!='application/json':return self.respond(b'{}',code=415)
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:4180','http://localhost:4180'):return self.respond(b'{}',code=403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>2048:raise ValueError('Invalid body size')
            body=json.loads(self.rfile.read(length));instruction=body['instruction']
            if not isinstance(instruction,str) or not 1<=len(instruction)<=500:raise ValueError('Instruction must be 1–500 characters')
            seed=body.get('seed')
            if seed is not None and (type(seed) is not int or not 0<=seed<=10000):raise ValueError('Invalid seed')
            mode=body.get('mode','baseline')
            if mode not in ['learned','baseline']:raise ValueError('Invalid mode')
        except Exception:return self.respond(b'{"error":"Invalid request"}',code=400)
        with lock:
            if state['running']:return self.respond(b'{"error":"Run already active"}',code=409)
            state.clear();state.update(running=True,stage='Starting',result=None,seed=seed,mode=mode)
        threading.Thread(target=job,args=(instruction,seed,mode),daemon=True).start()
        return self.respond(b'{"started":true}')

if __name__=='__main__':
    (ROOT/'artifacts').mkdir(exist_ok=True)
    print('TableTogether: http://127.0.0.1:4180',flush=True)
    ThreadingHTTPServer(('127.0.0.1',4180),Handler).serve_forever()
