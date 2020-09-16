import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.distributed import LangModelPipeline

groups = ['1']#['0','1','-1']
n_samples = 3
evaluator = Evaluator(n_samples,groups)


pipelines = []
contractions_actions = [False, True]#['off','passive','assertive']
for action in contractions_actions:
    pipelines.append(
    {'name': f'contractions_{action}',
      'pipe': LangModelPipeline,
      'params':{
              'model_name': 'distilbert-base-uncased',
              'topK_for_completions': 10000,
              'topK_for_biasing': 20,
              'split_sents': True,
              'contraction_action':action,
        }
     })

eval_data = evaluator.evaluate(pipelines)

#ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#with open('./results/'+ns+'.json', 'w') as f:
#    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
#res_frame.to_csv('./results/'+ns+'.csv',index=False)
