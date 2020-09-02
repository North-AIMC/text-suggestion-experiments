from pipelines.vocabs.words import words
import random

class BaselineCompletionsPipeline(object):
    def __init__(self):
        self.words = [s.lower() for s in words['neu']+words['neg']+words['pos']]
        # Could be any vocabulary!! Even the LM's!

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.words))
        elif(group=='-'):
            return(self._get_completions(text,self.words))
        else:
            if(random.sample(['+','-'], 1)=='+'):
                self._get_completions(text,self.words)
            else:
                self._get_completions(text,self.words)
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
            return(words[:3])

class BalancedBiasedCompletionsWithCapsPipeline(object):
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]

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
            return(words[:3])

class BiasedCompletionsWithCapsPipeline(object):
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
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
            return(words[:3])

class BiasedCompletionsPipeline(object):
    def __init__(self):
        self.pos_words = [s.lower() for s in words['pos']+words['neu']]
        self.neg_words = [s.lower() for s in words['neg']+words['neu']]

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(self._get_completions(text,self.pos_words))
        elif(group=='-'):
            return(self._get_completions(text,self.neg_words))
        else:
            return(['','',''])

    def _get_completions(self, text, words):
        if(text==''):
            return([s.capitalize() for s in words[:3]])
        elif(text[-1]!=' '):
            currentWord = text.rsplit(' ', 1)[-1]
            valid = [w for w in words if w.startswith(currentWord)]
            if(len(valid)<3):
                valid += ['']*(3 - len(valid))
            return(valid[:3])
        else:
            return(words[:3])
