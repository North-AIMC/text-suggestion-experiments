import nltk
import numpy
import torch
from transformers import BertTokenizer, BertForMaskedLM
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Prepare sentiment model
nltk.download('vader_lexicon')


# Define model object class
# INCLUDE THINGS WE HAVE LEARNED
class UpdatedOriginalPipeline(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        self.sid = SentimentIntensityAnalyzer()

    def get_topK_next_words(self, text, K):
        text += '<mask>.'
        text = text.replace('<mask>', self.tokenizer.mask_token)
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        mask_idx = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()[0]
        with torch.no_grad():
            output = self.model(input_ids)
        topk = self.tokenizer.decode(output[0][0,mask_idx,:].topk(K).indices.tolist()).split()
        return(topk)

    def rank_suggestions_by_sentiment(self, input_text, suggestions):
        text_suggestions = [input_text + s for s in suggestions]
        sentiment_scores = [self.sid.polarity_scores(s)['compound'] for s in text_suggestions]
        sentiment_ranking = numpy.argsort(sentiment_scores)
        ranked_suggestions = [suggestions[i] for i in sentiment_ranking]
        return(ranked_suggestions)

    def get_suggestions(self, input_text, sentiment_bias):
        if(input_text==''):
            suggestions = ['','','']
        elif(input_text[-1]==' '):
            # get top-K next words
            K = 20
            topK_suggestions = self.get_topK_next_words(input_text, K)

            # Rank suggestions by sentiment (negative to positive)
            ranked_suggestions = self.rank_suggestions_by_sentiment(input_text, topK_suggestions)

            # Bias suggestions based on sentiment
            if(sentiment_bias=='1'):
                suggestions = ranked_suggestions[::-1][:3]
            elif(sentiment_bias=='-1'):
                suggestions = ranked_suggestions[:3]
            else:
                suggestions = ['','','']
        else:
            input_text_split = input_text.split(' ')
            stem_text = ' '.join(input_text_split[:-1])+' '
            end_text = input_text_split[-1]

            # get top-K next words (using stem text)
            K = 10000
            topK_suggestions = self.get_topK_next_words(stem_text, K)

            # Filter to valid suggestions
            k = 20
            valid_suggestions = list(filter(lambda s: s.startswith(end_text), topK_suggestions))
            valid_topk = valid_suggestions[:k]

            # Rank suggestions by sentiment (negative to positive)
            ranked_suggestions = self.rank_suggestions_by_sentiment(stem_text, valid_topk)

            # Bias suggestions based on sentiment
            if(sentiment_bias=='1'):
                suggestions = ranked_suggestions[::-1][:3]
            elif(sentiment_bias=='-1'):
                suggestions = ranked_suggestions[:3]
            else:
                suggestions = ['','','']
        return(suggestions)

# Define model object class
class OriginalPipeline(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        self.sid = SentimentIntensityAnalyzer()

    def get_topK_next_words(self, input_text, K):
        input_text += '<mask>.'
        input_text = input_text.replace('<mask>', self.tokenizer.mask_token)
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        mask_idx = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()[0]
        with torch.no_grad():
            output = self.model(input_ids)
        topk = self.tokenizer.decode(output[0][0,mask_idx,:].topk(K).indices.tolist()).split()
        return(topk)

    def rank_suggestions_by_sentiment(self, input_text, suggestions):
        text_suggestions = [input_text + s for s in suggestions]
        sentiment_scores = [self.sid.polarity_scores(s)['compound'] for s in text_suggestions]
        sentiment_ranking = numpy.argsort(sentiment_scores)
        ranked_suggestions = [suggestions[i] for i in sentiment_ranking]
        return(ranked_suggestions)

    def get_suggestions(self, input_text, sentiment_bias):
        if(input_text==''):
            suggestions = ['','','']
        elif(input_text[-1]==' '):
            # get top-K next words
            K = 20
            topK_suggestions = self.get_topK_next_words(input_text, K)

            # Rank suggestions by sentiment (negative to positive)
            ranked_suggestions = self.rank_suggestions_by_sentiment(input_text, topK_suggestions)

            # Bias suggestions based on sentiment
            if(sentiment_bias=='1'):
                suggestions = ranked_suggestions[::-1][:3]
            elif(sentiment_bias=='-1'):
                suggestions = ranked_suggestions[:3]
            else:
                suggestions = ['','','']
        else:
            input_text_split = input_text.split(' ')
            stem_text = ' '.join(input_text_split[:-1])+' '
            end_text = input_text_split[-1]

            # get top-K next words (using stem text)
            K = 10000
            topK_suggestions = self.get_topK_next_words(stem_text, K)

            # Filter to valid suggestions
            k = 20
            valid_suggestions = list(filter(lambda s: s.startswith(end_text), topK_suggestions))
            valid_topk = valid_suggestions[:k]

            # Rank suggestions by sentiment (negative to positive)
            ranked_suggestions = self.rank_suggestions_by_sentiment(stem_text, valid_topk)

            # Bias suggestions based on sentiment
            if(sentiment_bias=='1'):
                suggestions = ranked_suggestions[::-1][:3]
            elif(sentiment_bias=='-1'):
                suggestions = ranked_suggestions[:3]
            else:
                suggestions = ['','','']
        return(suggestions)
