import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.tests import NullTestPipeline, BiasTestPipeline, TimeTestPipeline
from pipelines.completions import BiasedCompletionsPipeline, BiasedCompletionsWithCapsPipeline, BalancedBiasedCompletionsWithCapsPipeline
from pipelines.predictions import PredictionsPipeline, CompletionsWithPredictionsPipeline, CompletionsWithPredictionWithSentsPipeline

evaluator = Evaluator(10,['0','+','-'])
pipelines = [
    {'name': '1_NullTest',
     'pipe': NullTestPipeline},
    {'name': '2_BiasTest',
     'pipe': BiasTestPipeline},
    {'name': '3_TimeTest',
     'pipe': TimeTestPipeline},
    {'name': '4_BiasedCompletions',
     'pipe': BiasedCompletionsPipeline},
    {'name': '5_BiasedCompletionsWithCaps',
     'pipe': BiasedCompletionsWithCapsPipeline},
    {'name': '6_BalancedBiasedCompletionsWithCaps',
     'pipe': BalancedBiasedCompletionsWithCapsPipeline},
    {'name': '7_PredictionsPipeline',
     'pipe': PredictionsPipeline},
    {'name': '8_CompletionsWithPredictions',
     'pipe': CompletionsWithPredictionsPipeline},
    {'name': '9_CompletionsWithPredictionWithSents',
     'pipe':  CompletionsWithPredictionWithSentsPipeline},
]
eval_data = evaluator.evaluate(pipelines)

ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
with open('./results/'+ns+'.json', 'w') as f:
    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(30))
res_frame.to_csv('./results/'+ns+'.csv',index=False)
