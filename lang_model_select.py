import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.distributed import LangModelPipeline

groups = ['1']
n_samples = 10
evaluator = Evaluator(n_samples,groups)

pipelines = []
lang_models = [
    'distilbert-base-uncased',
    'bert-base-uncased',
    'bert-large-uncased',
    'distilroberta-base',
    'roberta-base',
    'xlm-roberta-base',
]
for i, model_name in enumerate(lang_models):
    pipelines.append(
    {'name': f'{i}_{model_name}',
      'pipe': LangModelPipeline,
      'params':{
              'model_name': model_name,
              'topK_for_completions': 10000,
              'topK_for_biasing': 3,
              'split_sents': True
        }
     })

eval_data = evaluator.evaluate(pipelines)
ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
with open('./results/lang_model_select'+ns+'.json', 'w') as f:
    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
res_frame.to_csv('./results/'+ns+'.csv',index=False)
