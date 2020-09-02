import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.originals import OriginalPipeline

evaluator = Evaluator(5,['0','1','-1'])
pipelines = [
    {'name': '10_Original',
      'pipe': OriginalPipeline},
]
eval_data = evaluator.evaluate(pipelines)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
