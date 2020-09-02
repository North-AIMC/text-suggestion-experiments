from numpy import argsort
from transformers import pipeline
from pipelines.vocabs.words import words
import random

class FasterCompletionsWithPredictionsPipeline(object): # This will become
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':10000,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text,self.pos_words)
            else:
                self._get_completions(text,self.neg_words)
            return(['','',''])

    def _get_completions(self, text, words):
        # If input text is empty
        if(text==''):
            return(self._format_valid_suggestions(words, True))
        elif(text[-1]!=' '):
            currentWord = text.rsplit(' ', 1)[-1]
            isCapitalised = False if currentWord=='' else currentWord[0].isupper()

            # If it is the first word
            if(len(text.lstrip().split(' '))==1):
                valid = [w for w in words if w.startswith(currentWord.lower())]
            else:
                words = self._predict_next_words(text.rsplit(' ', 1)[0].strip())
                valid = [w for w in words if w.startswith(currentWord.lower())]

            # Deal with capitalisation and formatting to ['','','']
            return(self._format_valid_suggestions(valid, isCapitalised))
        else:
            valid = self._predict_next_words(text)
            return(self._format_valid_suggestions(valid, False))

    def _predict_next_words(self, text):
        masked_text = f'{text.strip()} {self.mask}.'
        return([s['token_str'] for s in self.pipe(masked_text) if not s['token_str'].startswith('##')])

    def _format_valid_suggestions(self, valid, isCapitalised):
        if(isCapitalised):
            valid = [s.capitalize() for s in valid]
        if(len(valid)<3):
            valid += ['']*(3 - len(valid))
        return(valid[:3])

class LM10kCompletionsWithPredictionsPipeline(object): # This will become
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':10000,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text,self.pos_words)
            else:
                self._get_completions(text,self.neg_words)
            return(['','',''])

    def _get_completions(self, text, words):
        # If input text is empty
        if(text==''):
            return(self._format_valid_suggestions(words, True))
        elif(text[-1]!=' '):
            currentWord = text.rsplit(' ', 1)[-1]
            isCapitalised = False if currentWord=='' else currentWord[0].isupper()

            # If it is the first word
            if(len(text.lstrip().split(' '))==1):
                valid = [w for w in words if w.startswith(currentWord.lower())]
            else:
                words = self._predict_next_words(text.rsplit(' ', 1)[0].strip())
                valid = [w for w in words if w.startswith(currentWord.lower())]

            # Deal with capitalisation and formatting to ['','','']
            return(self._format_valid_suggestions(valid, isCapitalised))
        else:
            valid = self._predict_next_words(text)
            return(self._format_valid_suggestions(valid, False))

    def _predict_next_words(self, text):
        masked_text = f'{text.strip()} {self.mask}.'
        return([s['token_str'] for s in self.pipe(masked_text) if not s['token_str'].startswith('##')])

    def _format_valid_suggestions(self, valid, isCapitalised):
        if(isCapitalised):
            valid = [s.capitalize() for s in valid]
        if(len(valid)<3):
            valid += ['']*(3 - len(valid))
        return(valid[:3])


class LMCompletionsWithPredictionsPipeline(object): # This will become
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':1000,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text,self.pos_words)
            else:
                self._get_completions(text,self.neg_words)
            return(['','',''])

    def _get_completions(self, text, words):
        # If input text is empty
        if(text==''):
            return(self._format_valid_suggestions(words, True))
        elif(text[-1]!=' '):
            currentWord = text.rsplit(' ', 1)[-1]
            isCapitalised = False if currentWord=='' else currentWord[0].isupper()

            # If it is the first word
            if(len(text.lstrip().split(' '))==1):
                valid = [w for w in words if w.startswith(currentWord.lower())]
            else:
                words = self._predict_next_words(text.rsplit(' ', 1)[0].strip())
                valid = [w for w in words if w.startswith(currentWord.lower())]

            # Deal with capitalisation and formatting to ['','','']
            return(self._format_valid_suggestions(valid, isCapitalised))
        else:
            valid = self._predict_next_words(text)
            return(self._format_valid_suggestions(valid, False))

    def _predict_next_words(self, text):
        masked_text = f'{text.strip()} {self.mask}.'
        return([s['token_str'] for s in self.pipe(masked_text) if not s['token_str'].startswith('##')])

    def _format_valid_suggestions(self, valid, isCapitalised):
        if(isCapitalised):
            valid = [s.capitalize() for s in valid]
        if(len(valid)<3):
            valid += ['']*(3 - len(valid))
        return(valid[:3])


class PredictionsPipeline(object): # This will become
    def __init__(self):
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':20,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text))
        elif(group=='-'):
            return(self._get_completions(text))
        else:
            return(['','',''])

    def _get_completions(self, text):
        if(text==''):
            return(['','',''])
        elif(text[-1]!=' '):
            return(['','',''])
        else:
            masked_text = f'{text.strip()} {self.mask}.'
            valid = [s['token_str'] for s in self.pipe(masked_text) if not s['token_str'].startswith('##')]
            if(len(valid)<3):
                valid += ['']*(3 - len(valid))
            return(valid[:3])


class CompletionsWithPredictionsPipeline(object): # This will become
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':20,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text,self.pos_words)
            else:
                self._get_completions(text,self.neg_words)
            return(['','',''])

    def _get_completions(self, text, words):
        if(text==''):
            return([s.capitalize() for s in words[:3]])
        elif(text[-1]!=' '):
            currentWord = text.rsplit(' ', 1)[-1]
            isCapitalised = False if currentWord=='' else currentWord[0].isupper()
            valid = [w for w in words if w.startswith(currentWord.lower())]
            if(isCapitalised):
                valid = [s.capitalize() for s in valid]
            if(len(valid)<3):
                valid += ['']*(3 - len(valid))
            return(valid[:3])
        else:
            masked_text = f'{text.strip()} {self.mask}.'
            valid = [s['token_str'] for s in self.pipe(masked_text) if not s['token_str'].startswith('##')]
            if(len(valid)<3):
                valid += ['']*(3 - len(valid))
            return(valid[:3])

class CompletionsWithPredictionWithSentsPipeline(object): # This will become
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]
        pipe_spec = {
            'model': 'distilbert-base-uncased',
            'topk':20,
        }
        self.pipe = pipeline('fill-mask', **pipe_spec)
        self.mask = self.pipe.tokenizer.mask_token

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text,self.pos_words)
            else:
                self._get_completions(text,self.neg_words)
            return(['','',''])

    def _get_completions(self, text, words):
        if(text==''):
            return([s.capitalize() for s in words[:3]])
        elif(text[-1]!=' '):
            currentWord = text.rsplit(' ', 1)[-1]
            isCapitalised = False if currentWord=='' else currentWord[0].isupper()
            valid = [w for w in words if w.startswith(currentWord.lower())]
            if(isCapitalised):
                valid = [s.capitalize() for s in valid]
            if(len(valid)<3):
                valid += ['']*(3 - len(valid))
            return(valid[:3])
        else:
            text = text.rsplit('.', 1)[-1].lstrip()
            if(text==''):
                return([s.capitalize() for s in words[:3]])
            else:
                masked_text = f'{text.strip()} {self.mask}.'
                valid = [s['token_str'] for s in self.pipe(masked_text) if not s['token_str'].startswith('##')]
                if(len(valid)<3):
                    valid += ['']*(3 - len(valid))
                return(valid[:3])
