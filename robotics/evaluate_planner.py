"""Small semantic check of the actual OpenVINO model, including unsupported requests."""
import json
from planner import Planner
from scene import ROOT
planner=Planner()
cases=[('Set the table for one person.',True),
       ('Place the cup to the right of the plate, and the utensil on the left.',True),
       ('Put the cup on the left of the plate.',False),
       ('Stack three plates on top of each other.',False)]
results=[]
for instruction,expected in cases:
    try:
        plan=planner.plan(instruction)
        result={'instruction':instruction,'accepted':True,'plan':plan}
    except Exception as exc:result={'instruction':instruction,'accepted':False,'reason':str(exc)}
    result.update(expected_acceptance=expected,passed=result['accepted']==expected)
    results.append(result)
report={'passed':sum(r['passed'] for r in results),'total':len(results),'results':results}
(ROOT/'artifacts/planner-evaluation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
