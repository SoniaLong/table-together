"""OpenVINO language planner with an explicit bounded task contract."""
import json
import time
from pathlib import Path
import openvino_genai as genai

class Planner:
    def __init__(self,device='CPU',model=None):
        model=Path(model) if model else Path(__file__).resolve().parents[1]/'models/qwen'
        self.pipe=genai.LLMPipeline(str(model),device)
        self.device=device

    def plan(self,instruction):
        system='''Translate a table-setting request into JSON only. The scene has one plate, one cup, one utensil.
Supported task: put cup on the right of the plate and utensil on the left. Return {"task":"set_table"}.
An instruction mentioning the utensil on the LEFT is supported. Only a CUP on the LEFT is unsupported.
Accept synonymous verbs: set, arrange, place, put, prepare. The user need not say 'one person'.
If the user asks for anything else or another layout, return {"task":"unsupported"}.
Never output coordinates or code.'''
        examples='''Request: Set the table for one person.
JSON: {"task":"set_table"}
Request: Put the cup to the right of the plate and the utensil to its left.
JSON: {"task":"set_table"}
Request: Arrange a place setting with cutlery on the left and a drinking cup on the right.
JSON: {"task":"set_table"}
Request: Cup: right. Utensil: left.
JSON: {"task":"set_table"}
Request: Stack the cups.
JSON: {"task":"unsupported"}
Request: Put the cup to the left of the plate.
JSON: {"task":"unsupported"}
'''
        prompt=f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{examples}Request: {instruction}\nJSON:<|im_end|>\n<|im_start|>assistant\n'
        start=time.perf_counter()
        raw=str(self.pipe.generate(prompt,max_new_tokens=32,do_sample=False,apply_chat_template=False)).strip()
        parsed=json.loads(raw)
        if parsed!={'task':'set_table'}:raise ValueError('The model did not select the supported one-person layout; no movement executed.')
        return {**parsed,'instruction':instruction,'device':self.device,'inference_ms':(time.perf_counter()-start)*1000,'raw_output':raw,'model':'Qwen2.5-Coder-0.5B-Instruct INT4'}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('instruction');parser.add_argument('--device',default='CPU');args=parser.parse_args()
    print(json.dumps(Planner(args.device).plan(args.instruction),indent=2))
