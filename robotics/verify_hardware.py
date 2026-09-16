"""Save actual device/inference evidence. Never infer contest eligibility from a GPU name."""
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import openvino as ov
from openvino import opset13 as ops

def verify():
    cpu = platform.processor()
    if platform.system() == 'Windows':
        cpu = subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],text=True).strip()
    elif Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),cpu)
    core = ov.Core()
    report = {'cpu':cpu,'os':platform.platform(),'openvino':ov.__version__,
              'eligibility':'UNVERIFIED: check CPU against Intel Series 2/3 catalog and organizer remote-run rules',
              'devices':{}}
    x = ops.parameter([1,16], np.float32)
    model = ov.Model([ops.relu(x)], [x])
    for device in core.available_devices:
        result = {'name':core.get_property(device,'FULL_DEVICE_NAME')}
        try:
            compiled = core.compile_model(model,device)
            start=time.perf_counter()
            output=compiled([np.ones((1,16),dtype=np.float32)])[0]
            result.update(smoke_inference_pass=bool(np.allclose(output,1)),latency_ms=(time.perf_counter()-start)*1000)
        except Exception as exc: result['error']=str(exc)
        report['devices'][device]=result
    return report

if __name__ == '__main__':
    report=verify()
    out=Path(__file__).resolve().parent/'artifacts'
    out.mkdir(exist_ok=True)
    (out/'hardware.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
