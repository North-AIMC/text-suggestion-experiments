import re
import torch
import numpy
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pipelines.vocabs.words import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class LangModelPipeline(object):
    def __init__(self, model_name, topK_for_completions, topK_for_biasing, split_sents): # Make cleaner (!!)
        # Params
        self.model_name = model_name
        self.topK_for_completions = topK_for_completions
        self.topK_for_biasing = topK_for_biasing
        self.split_sents = split_sents

        # Load tokenizer and language model (easy to change model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).eval()

        # Load first word suggestions (easy to change vocabulary and ranking)
        self.pos_words = [s.lower() for s in words['neu']+words['pos']]
        self.neg_words = [s.lower() for s in words['neu']+words['neg']]

        # Pre-compute sentiment scores(???)
        self.scorer = SentimentIntensityAnalyzer()


    def get_suggestions(self, text, group):
        # Simpler particpant router,
        # mapping group labels to something more meaningful
        if(group=='1'):
            return(self._get_valid_suggestions(text, self.pos_words, 1))
        elif(group=='-1'):
            return(self._get_valid_suggestions(text, self.neg_words, -1))
        else:
            if(random.sample(['1','-1'], 1)=='1'):
                self._get_valid_suggestions(text, self.pos_words, 1)
            else:
                self._get_valid_suggestions(text, self.neg_words, -1)
            return(['','',''])

    def _get_valid_suggestions(self, text, first_words, sentiment_bias):
        # If specified, only use last sentence
        if(self.split_sents):
            text = re.split(r'[.!?]', text)[-1]

        # If nothing has been typed, return default suggestions
        if(not text.strip()):
            return(self._format_valid_suggestions(first_words, True))

        # If in the middle of a word, get completions
        elif(text[-1].isalpha()):
            return(self._get_completions(text, first_words, sentiment_bias))

        # If starting a new word, get predictions
        elif(text[-1] in ' ,;:'):
            # Get predictions
            predictions = self._get_predictions(text, self.topK_for_biasing)

            # Bias predictions
            ranked = self._bias_suggestions(predictions, sentiment_bias)

            # Format and return
            return(self._format_valid_suggestions(ranked, False))

        # Otherwise (e.g. numbers), return nothing
        else:
            return(['','',''])

    def _get_completions(self, text, first_words, sentiment_bias):
        # If first word, then use first_words for completions
        if(len(text.lstrip().split(' '))==1):
            # Get current word
            currentWord = text.rsplit(' ', 1)[-1]

            # Identify valid completions, format, and return
            valid = [w for w in first_words if w.startswith(currentWord.lower())]
            return(self._format_valid_suggestions(valid, currentWord[0].isupper()))

        # Else, use predictions for completions
        else:
            # Predict vocabulary for completions
            predictions = self._get_predictions(text, self.topK_for_completions)


            # Get current word
            currentWord = text.rsplit(' ', 1)[-1]

            # Get valid predictions
            valid = [w for w in predictions if w.startswith(currentWord.lower())]

            # Get topK_for_biasing and rank by sentiment (could be cleaned up)
            ranked = self._bias_suggestions(valid[:self.topK_for_biasing], sentiment_bias)

            # Format and return
            return(self._format_valid_suggestions(ranked, currentWord[0].isupper()))

    def _bias_suggestions(self, suggestions, sentiment_bias):
        scores = [-sentiment_bias * self.scorer.polarity_scores(s)['compound'] for s in suggestions]
        return([suggestions[i] for i in numpy.argsort(scores)])


    def _get_predictions(self, text, topK):
        # Trim off current word and add on mask token and .
        text = text.rsplit(' ', 1)[0].strip()+f' {self.tokenizer.mask_token}.'

        # Get input_ids for tokens
        input_ids = self.tokenizer.encode(text, return_tensors='pt')

        # Find mask ID
        mask_idx = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()[0]

        # Forward pass through language model
        with torch.no_grad():
            output = self.model(input_ids)

        # Get topK_ids for next word
        topK_ids = output[0][0,mask_idx,:].topk(topK).indices.tolist()

        # Decode back tokens (NB: Make ids_to_tokens...)
        return(self.tokenizer.decode(topK_ids).split())


    def _format_valid_suggestions(self, valid, isCapitalised):
        # Ensure 3, and only 3, suggestions are returned
        if(len(valid)<3):
            valid += ['']*(3 - len(valid))
        else:
            valid = valid[:3]

        # Ensure suggestions are capitalized correctly
        if(isCapitalised):
            valid = [s.capitalize() for s in valid]
        return(valid)
