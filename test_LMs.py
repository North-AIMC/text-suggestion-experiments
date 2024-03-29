import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.completions import BaselineCompletionsPipeline

evaluator = Evaluator(10,['0','+','-'])
pipelines = [
     {'name': 'BaselineCompletions',
      'pipe': BaselineCompletionsPipeline}
]
eval_data = evaluator.evaluate(pipelines)

# Move saving into evaluator!!
#ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#with open('./results/'+ns+'.json', 'w') as f:
#    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
#res_frame.to_csv('./results/'+ns+'.csv',index=False)
