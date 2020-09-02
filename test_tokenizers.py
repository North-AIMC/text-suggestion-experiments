# Test different tokenizers on test data
# By encoding, then decoding, the comparing to the original
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).eval()

input_text = "This is stupid, don't do it!!!"
print(input_text)
print(tokenizer.tokenize(input_text))

# Iterate through tokens and predict the next


#encoded_input = tokenizer(input_text, add_special_tokens=False)
#print(encoded_input['input_ids'])
#print(tokenizer.convert_ids_to_tokens(encoded_input['input_ids']))
#print(tokenizer.decode(encoded_input['input_ids']))
