import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.completions import BasicCompletionPipeline, CompletionWithCapsPipeline

evaluator = Evaluator(10,['0','+','-'])
pipelines = [
     {'name': 'BasicCompletionPipeline',
      'pipe': BasicCompletionPipeline},
      {'name': 'CompletionWithCapsPipeline',
       'pipe': CompletionWithCapsPipeline}
]
eval_data = evaluator.evaluate(pipelines)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
