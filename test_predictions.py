import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.predictions import LM10kCompletionsWithPredictionsPipeline, FasterCompletionsWithPredictionsPipeline

evaluator = Evaluator(5,['0','+','-'])
pipelines = [
    {'name': '3_LM10kCompletionsWithPredictions',
      'pipe': LM10kCompletionsWithPredictionsPipeline},
    {'name': '4_FasterCompletionsWithPredictions',
      'pipe': FasterCompletionsWithPredictionsPipeline},
]
eval_data = evaluator.evaluate(pipelines)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
