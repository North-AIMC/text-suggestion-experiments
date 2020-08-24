"""
This is for comparing language models on word suggestion:
- Only make suggestions after spaces.
- Autocomplete test are elsewhere
"""

import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.LMs import BasicLMPipeline, SuggestionLMPipeline

evaluator = Evaluator(10,['0','+','-'])
pipelines = [
     {'name': 'SuggestionLMPipeline',
      'pipe': SuggestionLMPipeline}
]
eval_data = evaluator.evaluate(pipelines)

# Move saving into evaluator!!
#ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#with open('./results/'+ns+'.json', 'w') as f:
#    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
#res_frame.to_csv('./results/'+ns+'.csv',index=False)
