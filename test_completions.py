import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.completions import BiasedCompletionsPipeline, BiasedCompletionsWithCapsPipeline, BalancedBiasedCompletionsWithCapsPipeline

evaluator = Evaluator(5,['0','+','-'])
pipelines = [
     {'name': '4_BiasedCompletions',
      'pipe': BiasedCompletionsPipeline},
     {'name': '5_BiasedCompletionsWithCaps',
      'pipe': BiasedCompletionsWithCapsPipeline},
     {'name': '6_BalancedBiasedCompletionsWithCaps',
      'pipe': BalancedBiasedCompletionsWithCapsPipeline},
]
eval_data = evaluator.evaluate(pipelines)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
