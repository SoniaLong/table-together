"""Run the real language-to-manipulation pipeline with OpenVINO explicitly on CPU."""
import json
import time
from datetime import datetime, timezone
import openvino as ov
import mujoco
from scene import ROOT
from planner import Planner
from episode import run

def main():
    out=ROOT/'artifacts/local-cpu-check'
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'cpu':ov.Core().get_property('CPU','FULL_DEVICE_NAME'),
            'openvino':ov.__version__,'mujoco':mujoco.__version__,
            'inference_device':'CPU','physics':'Standard MuJoCo CPU simulation',
            'rendering':'OpenGL camera rendering; software-only rendering was not forced',
            'seed':0}
    try:
        planner=Planner(device='CPU')
        report['model_load_seconds']=time.perf_counter()-started
        report['plan']=planner.plan('Set the table for one person. Put the cup to the right of the plate and the utensil to the left.')
        sim_start=time.perf_counter()
        events=run(record_video=False,seed=0,out_dir=out)
        report['simulation_and_camera_wall_seconds']=time.perf_counter()-sim_start
        report['placements']=[event for event in events if 'evaluation' in event]
        report['final_checks']=events[-1]
        report['passed']=bool(events[-1]['success'])
    except Exception as exc:
        report.update(passed=False,error=str(exc))
    report['total_wall_seconds']=time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
