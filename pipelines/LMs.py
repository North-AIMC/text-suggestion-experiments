from numpy import argsort
from transformers import pipeline

class SuggestionLMPipeline(object):
    def __init__(self):
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':1000,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(text==''):
            return(['','',''])
        elif(text[-1]==' '):
            masked_text = f'{text.strip()} {self.mask}.'
            suggestions = [s['token_str'] for s in self.pipe(masked_text)]
            if(group=='+'):
                return(suggestions)
            elif(group=='-'):
                return(suggestions)
            else:
                return(['','',''])
        else:
            return(['','',''])

class BasicLMPipeline(object):
    def __init__(self):
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':5,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        masked_text = f'{text.strip()} {self.mask}.'
        suggestions = [s['token_str'] for s in self.pipe(masked_text)[:3]]
        if(group=='+'):
            return(suggestions)
        elif(group=='-'):
            return(suggestions)
        else:
            return(['','',''])
