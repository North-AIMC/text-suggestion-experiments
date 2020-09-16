# Test deciding and thresholding
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load tokenizer and language model (easy to change model)
#model_name = 'roberta-base'
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).eval()

def _get_predictions(text, topK):
    # Trim off current word and add on mask token and .
    text = text.rsplit(' ', 1)[0].strip()+f' {tokenizer.mask_token}.'

    # Get input_ids for tokens
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Find mask ID
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    # Forward pass through language model
    with torch.no_grad():
        output = model(input_ids)

    # Need to generate predictions and values
    # Need to use topK still, but then threshold
    #values, predictions = output[0][0,mask_idx,:].softmax(dim=0).topk(topK)
    #topK_tokens = tokenizer.convert_ids_to_tokens(predictions.tolist(), skip_special_tokens=True)
    #topK_valid = [t.replace('Ġ','') for t in topK_tokens if t.startswith('Ġ')]
    #return(topK_valid)

    # Get topK_ids for next word
    topK_ids = output[0][0,mask_idx,:].topk(topK).indices.tolist()
    valid = [t for t in tokenizer.convert_ids_to_tokens(topK_ids) if not t.startswith('##')]

    # Decode back tokens (NB: Make ids_to_tokens...)
    return(valid)

text = 'This is '
print(_get_predictions(text, 100))
