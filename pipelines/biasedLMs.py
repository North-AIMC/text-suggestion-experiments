from numpy import argsort
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
download('vader_lexicon')

class BiasedLMPipeline(object):
    def __init__(self):
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':10,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token
        self.sid = SentimentIntensityAnalyzer()

    def get_suggestions(self, text, group):
        masked_text = f'{text.strip()} {self.mask}.'
        suggestions = [s['token_str'] for s in self.pipe(masked_text)]
        sentiments = [self.sid.polarity_scores(s)['compound'] for s in suggestions]
        ranking = argsort(sentiments)
        ranked_suggestions = [suggestions[i] for i in ranking]
        if(group=='+'):
            return(ranked_suggestions[::-1][:3])
        elif(group=='-'):
            return(ranked_suggestions[:3])
        else:
            return(['','',''])
