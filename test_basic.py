"""
Just for testing things are working properly...
"""
import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.tests import NullPipeline, BiasTestPipeline, StopWordTestPipeline

evaluator = Evaluator(10,['0','+','-'])
pipelines = [
    {'name': 'NullPipeline',
     'pipe': NullPipeline},
    {'name': 'BiasTestPipeline',
     'pipe': BiasTestPipeline},
    {'name': 'StopWordTestPipeline',
     'pipe': StopWordTestPipeline}
]
eval_data = evaluator.evaluate(pipelines)

ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
with open('./results/'+ns+'.json', 'w') as f:
    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
res_frame.to_csv('./results/'+ns+'.csv',index=False)
