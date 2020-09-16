from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import json
import random

# Create your own LineByLineDataset
def _get_texts(num_samples):
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
    aggregated_samples = [s.replace('\n',' ') for s in list(set(aggregated_samples)) if s!='']
    return(random.sample(aggregated_samples, num_samples))

print(_get_texts(10))
