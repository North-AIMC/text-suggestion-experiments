import random
import string
import json
import pandas as pd

import nltk

from evaluation import Evaluator
from pipelines.distributed import LangModelPipeline

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Specify experimental parameters
groups = ['0','1','-1']
num_user_per_group = 5
num_task_per_user = 10


# Sample target_texts
n_samples = len(groups)*num_user_per_group*num_task_per_user
evaluator = Evaluator(n_samples,groups)
target_texts = evaluator._get_texts(n_samples)

# Setup text suggestion pipeline
pipe_params = {
        'model_name': 'roberta-base',
        'topK_for_completions': 10000,
        'topK_for_biasing': 10,
        'split_sents': True
        }
pipe = LangModelPipeline(**pipe_params)

def calculate_outcomes(target_text, path_data):
    total_chars = len(target_text)
    total_clicks = len(path_data)
    total_time = sum([d['request_time'] for d in path_data])
    total_bias = sum([sum([sid.polarity_scores(s)['compound'] for s in d['suggestions']]) for d in path_data])

    clicks_per_chars = total_clicks / total_chars
    bias_per_suggestion = total_bias / (3*total_clicks)
    ms_per_request = 1000*total_time / total_clicks
    return({
        'total_chars': total_chars,
        'total_clicks': total_clicks,
        'total_time': total_time,
        'total_bias': total_bias,
        'clicks_per_chars': clicks_per_chars,
        'bias_per_suggestion': bias_per_suggestion,
        'ms_per_request': ms_per_request,
    })


# Allocate target texts to users and users to groups
test_data_grouped = []
for group in groups:
    for i in range(num_user_per_group):
        # Generate a (fake) proflific ID and pop num_task_per_user replies from target_texts
        user_replies, target_texts = target_texts[:num_task_per_user], target_texts[num_task_per_user:]

        # Could analyse the results here (create a batch version of evaluate!)
        outcomes = {}
        outcomes['user_id'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        outcomes['group_id'] =  group
        for j, target_text in enumerate(user_replies):
            path_data = evaluator._simulate(pipe, group, target_text)
            outcomes[f'task_{j}'] = calculate_outcomes(target_text, path_data)['bias_per_suggestion']


        # Log results
        print(f'{group}_{i}')
        test_data_grouped.append(outcomes)

print(json.dumps(test_data_grouped, indent=4))

df = pd.DataFrame(test_data_grouped)
df.to_csv('./results/proxy_outcome_data.csv',index=False)
