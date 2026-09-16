"""Validate the portable graph on this host; never emulate target-hardware evidence."""
from pathlib import Path
import json,platform,time
import numpy as np
import openvino as ov
from policy import tokenize,INSTRUCTIONS

root=Path(__file__).resolve().parent
core=ov.Core();model=core.read_model(root/'checkpoints/policy-fp32.xml')
supported=core.query_model(model,'CPU')
compiled=core.compile_model(model,'CPU',{'INFERENCE_PRECISION_HINT':'f32'})
inputs={'image':np.zeros((1,3,96,128),np.float32),'state':np.zeros((1,12),np.float32),'tokens':tokenize(INSTRUCTIONS[0])[None,:],'step':np.array([0],np.int64)}
result=compiled(inputs)
report={'os':platform.system(),'architecture':platform.machine(),'actual_processor':core.get_property('CPU','FULL_DEVICE_NAME'),'openvino':ov.__version__,'cpu_query_supported_operations':len(supported),'graph_operations':len(model.get_ordered_ops()),'output_shape':list(result['actions'].shape),'finite_actions':bool(np.isfinite(result['actions']).all()),'target_series_2_3_execution_verified':False,'note':'This report records this host only. Series 2/3 CPU compatibility is inferred separately from the Intel support matrix. Verify the actual target processor before asserting qualification.'}
(root/'checkpoints/compatibility.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
