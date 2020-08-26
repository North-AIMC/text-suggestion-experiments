import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.predictions import PredictionsPipeline, CompletionsWithPredictionsPipeline, CompletionsWithPredictionWithSentsPipeline

evaluator = Evaluator(30,['0','+','-'])
pipelines = [
     {'name': '7_PredictionsPipeline',
      'pipe': PredictionsPipeline},
     {'name': '8_CompletionsWithPredictions',
      'pipe': CompletionsWithPredictionsPipeline},
     {'name': '9_CompletionsWithPredictionWithSents',
      'pipe':  CompletionsWithPredictionWithSentsPipeline},
]
eval_data = evaluator.evaluate(pipelines)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
