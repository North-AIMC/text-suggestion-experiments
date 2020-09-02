from numpy import argsort
from transformers import pipeline
from pipelines.vocabs.words import words

class LangModelCompletionsPipeline(object):
    def __init__(self):
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':10000, # Needs to be tuned
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token
        self.coldstart = ['The','I','This']
        self.firstwords = words

    def get_suggestions(self, text, group):
        # Simple router-layer to: (maybe make 'router' a function...)
        # - balance latency across groups
        # - map group labels to something more numerical
        # - final tweaking of suggestions before sending them back
        if(group=='+'):
            return(self._get_completions(text, '+')) # Use the sentiment bias!
        elif(group=='-'):
            return(self._get_completions(text, '-'))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text, '+')
            else:
                self._get_completions(text, '-')
            return(['','',''])

    def _get_completions(self, text, bias):
        # If there are no characters in the input text
        if(not text.lstrip()):
            # Return "coldstart" suggestions
            return(self.coldstart)
        # Trim to current sentence
        text = text.rsplit('.', 1)[-1]
        # If there are no characters in current sentence
        if(not text.lstrip()):
            # Return "coldstart" suggestions
            return(self.coldstart)
        # If the last character is alphabetical
        if(text[-1].isalpha()):
            # If current word is only word
            if(len(text.lstrip().split(' '))==1):
                # Use FirstWordCompleter







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
