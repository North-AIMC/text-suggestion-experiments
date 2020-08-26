"""
Just for testing things are working properly...
"""
import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.tests import NullTestPipeline, BiasTestPipeline, TimeTestPipeline

evaluator = Evaluator(10,['0','+','-'])
pipelines = [
    {'name': '1_NullTest',
     'pipe': NullTestPipeline},
    {'name': '2_BiasTest',
     'pipe': BiasTestPipeline},
    {'name': '3_TimeTest',
     'pipe': TimeTestPipeline}
]
eval_data = evaluator.evaluate(pipelines)

#ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#with open('./results/'+ns+'.json', 'w') as f:
#    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
#res_frame.to_csv('./results/'+ns+'.csv',index=False)
