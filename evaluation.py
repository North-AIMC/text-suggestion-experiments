import json
import time
import random
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

class Evaluator(object):
    def __init__(self, num_samples, groups):
        self.num_samples = num_samples
        self.texts = [t for t in self._get_texts(num_samples)] # Lower case!
        self.groups = groups

    def evaluate(self, pipelines):
        results = []
        for pipeline in pipelines:

            # Initialise the pipeline
            if('params' in pipeline):
                print("Using params!")
                pipe = pipeline['pipe'](**pipeline['params'])
            else:
                pipe = pipeline['pipe']()
            print(pipeline['name'])

            # Iterate through the groups
            for group in self.groups:
                print(group)

                # Iterate through texts
                for i, target_text in enumerate(self.texts):
                    print(f'{i}/{self.num_samples}')

                    path_data = self._simulate(pipe, group, target_text)

                    results.append({
                        'pipe': pipeline['name'],
                        'group': group,
                        'target': target_text,
                        'path_data': path_data,
                    })
        return(results)

    def _simulate(self, pipe, group, target_text):
        # Setup simulation
        text_box = TextBox()
        user = User(target_text)
        # Run simulation
        path_data = []
        while len(text_box.current_text) < len(target_text):

            # Suggestions arrive
            start = time.time()
            print(text_box.current_text)
            suggestions = pipe.get_suggestions(text_box.current_text,group)
            print(suggestions)
            request_time = (time.time()-start)

            # User views current text and suggestions, and decides optimal key
            user.view_current_text(text_box.current_text)
            optimal_key_press = user.decide_optimal_key_press(suggestions[:3]) # User only sees 3 suggestions
            text_box.update_text(optimal_key_press)

            # Log
            path_data.append({
                'request_time': request_time,
                'suggestions': suggestions,
                'key_press': optimal_key_press,
            })
        return(path_data)


    def analyse(self, results):
        df = pd.DataFrame(results)

        def calcTotalBias(l):
            return(sum([sum([sid.polarity_scores(s)['compound'] for s in d['suggestions']]) for d in l]))

        # CALULATE METRICS
        df['total_chars'] = df['target'].apply(lambda s: len(s))
        df['total_clicks'] = df['path_data'].apply(lambda l: len(l))
        df['total_time'] = df['path_data'].apply(lambda l: sum([d['request_time'] for d in l]))
        df['total_time'] = df['path_data'].apply(lambda l: sum([d['request_time'] for d in l]))
        df['total_bias'] = df['path_data'].apply(calcTotalBias)

        # GROUPBY AND SUM
        df_pooled = df.groupby(['pipe','group']).agg({
            'total_chars': 'sum',
            'total_clicks': 'sum',
            'total_time': 'sum',
            'total_bias': 'sum'
        }).reset_index()

        # CALULATE POOLED METRICS
        df_pooled['clicks_per_chars'] = df_pooled['total_clicks'] / df_pooled['total_chars']
        df_pooled['bias_per_suggestion'] = df_pooled['total_bias'] / (3*df_pooled['total_clicks'])
        df_pooled['ms_per_request'] = 1000*(df_pooled['total_time'] / df_pooled['total_clicks'])
        return(df_pooled)


    def _get_texts(self,num_samples):
        pathToSamples = './data/'
        listOfSampleNames = [
            'BBCBreaking-2020-08-09-07-24-26_filtered.json',
            'BBCBreaking-2020-08-09-11-52-24_filtered.json',
            'BBCNews-2020-08-10-11-39-48_filtered.json',
            'BBCNews-2020-08-14-06-53-06_filtered.json',
            'BBCNews-2020-08-19-10-04-25_filtered.json',
            'SkyNews-2020-08-19-10-21-49_filtered.json',
            'BBCBreaking-2020-08-19-10-38-00_filtered.json',
        ]

        # Load an aggregate to find unique stings
        aggregated_samples = []
        for sampleName in listOfSampleNames:
            with open(pathToSamples+sampleName, 'r') as f:
                aggregated_samples.extend(json.load(f))

        # Get unique strings
        aggregated_samples = [s for s in list(set(aggregated_samples)) if s!='']
        return(random.sample(aggregated_samples, num_samples))

class TextBox(object):
    def __init__(self):
        self.current_text = ''
        self.puncuation = '.,!?;'

    def replace_text(self, text):
        self.current_text = text

    def update_text(self, key_press):
        self.current_text = self.calculate_update(key_press)

    def calculate_update(self, key_press):
        if(key_press['type']=='c'):
            return(self.current_text + key_press['value'])
        elif(key_press['type']=='s'):
            if(self.current_text==''):
                return(self.current_text + key_press['value'])
            else:
                last_character = self.current_text[-1]
                if(last_character==' '):
                    return(self.current_text + key_press['value'])
                elif(last_character in self.puncuation):
                    return(self.current_text + ' ' + key_press['value'])
                else:
                    trimmed_text = ''.join(self.current_text.rsplit(' ', 1)[:-1])
                    if(trimmed_text==''):
                        return(key_press['value'])
                    else:
                        return(trimmed_text + ' ' + key_press['value'])
        else:
            print('Unknown key press value!')

    def print_text(self):
        print(self.current_text)

class User(object):
    def __init__(self, target_text):
        self.target_text = target_text
        self.text_box = TextBox()

    def view_current_text(self, current_text):
        self.text_box.replace_text(current_text)

    def decide_optimal_key_press(self, suggestions):
        # Identify key-press options
        next_char = self.target_text[len(self.text_box.current_text)]
        key_press_options = [{'type':'c','value':next_char}]
        key_press_options += [{'type':'s','value':s} for s in suggestions]

        # Calculate validity and length for each option
        for key_press in key_press_options:
            update = self.text_box.calculate_update(key_press)
            key_press['update_valid'] = self.target_text.startswith(update)
            key_press['update_length'] = len(update)

        # Filter to valid options and idenitfy key press with longest valid update
        valid_key_press_options = [key_press for key_press in key_press_options if key_press['update_valid']]
        optimal_key_press = sorted(valid_key_press_options, key=lambda k: k['update_length'], reverse=True)[0]
        key_press = {'type': optimal_key_press['type'], 'value': optimal_key_press['value']}
        return(key_press)

# TextBox model (User class has an internal model):
# - If key-press is a character, then add character
# - If key-press is suggestion, and last character is space, then add suggestion
# - If key-press is suggestion, and last character is punctionation, then add space, then add suggestion
# - If key-press is suggestion, and last character is not space or punctuation, then replace it
# NB: Phone suggestion selections are normally followed by a space, but we ignore those here
# Model user doesn't do backspace, just takes shortcuts forward

# User model:
# - Has a 'target_text' they are trying to write
# - Recieves a 'current_text' from the 'text_box' (Need a text box class)
# - Recieves 'suggestions' from the 'text_rec'
# - Decides what key to press
# - Sends a 'key_value' to the 'text_box'
# DECIDE
        # - Identify correct suggestion
        # - Looks at suggestions
        # - Checks if target word is in there
        # - Check if a prefix of the target word is in there
        # - Else presses next character of target word
