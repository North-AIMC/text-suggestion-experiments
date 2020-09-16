
import json
from datetime import datetime
from evaluation import Evaluator
from pipelines.distributed import LangModelPipeline

groups = ['0','1','-1']
n_samples = 10
evaluator = Evaluator(n_samples,groups)

pipelines = []
for i in [10, 20]:
    pipelines.append(
    {'name': f'{i}_bias_topK_{i}',
      'pipe': LangModelPipeline,
      'params':{
              'model_name': 'distilbert-base-uncased',
              'topK_for_completions': 10000,
              'topK_for_biasing': i,
              'split_sents': True
        }
     })


eval_data = evaluator.evaluate(pipelines)

#ns = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#with open('./results/'+ns+'.json', 'w') as f:
#    json.dump(eval_data,f)

res_frame = evaluator.analyse(eval_data)
print(res_frame.head(20))
#res_frame.to_csv('./results/'+ns+'.csv',index=False)




"""
pipe = LangModelPipeline('distilbert-base-uncased')
print('\nTest _get_predictions:')
text = 'This is '
print(text)
print(pipe._get_predictions(text, 10))


print('\nTest _bias_suggestions:')
suggestions = ['the','hate','love']
print(pipe._bias_suggestions(suggestions, 1))
print(pipe._bias_suggestions(suggestions, -1))


#print('\nTest get_suggestions:')
target = 'This. Why, not?'

for i in range(len(target)):
    text = target[:i]
    print('------')
    print([text])
    print(pipe.get_suggestions(text, '1'))
    print(pipe.get_suggestions(text, '-1'))

"""
