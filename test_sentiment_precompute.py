print('This will test the speed of pre-computing sentiments for a given vocabulary...')
import time
from transformers import AutoTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

vocab = tokenizer.get_vocab()

def _precompute_scores(vocab):
    scorer = SentimentIntensityAnalyzer()
    return({word:scorer.polarity_scores(word)['compound'] for word in vocab.keys()})

print(_precompute_scores(vocab))

"""
print('Online computation:')
start = time.time()
for word, index in vocab.items():
    score = scorer.polarity_scores(word)['compound']
print((time.time()-start))

print('Pre-computed:')
start = time.time()
for word, index in vocab.items():
    score = scores[word]
print((time.time()-start))
"""
