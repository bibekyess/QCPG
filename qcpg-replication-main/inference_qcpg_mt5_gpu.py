import torch
from transformers import MT5Tokenizer, MT5Config, MT5ForConditionalGeneration

# FIXME
T5_PATH = 'bibekyess/qcpg-parabk2-mt5'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

t5_tokenizer = MT5Tokenizer.from_pretrained(T5_PATH)
t5_config = MT5Config.from_pretrained(T5_PATH)
t5_mlm = MT5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Input text
"""
    Here, the following sentence doesn't work:
    'COND_SEMANTIC_SIM_80 COND_LEXICAL_DIV_30 COND_SYNTACTIC_DIV_50Is this going to work or what are we doing here?'
    It is because this encoder doesn't accept 'underscore' or '_' infront of number
"""
text = 'Is this going to work or what are we doing here?'

encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

outputs = t5_mlm.generate(input_ids=input_ids)

def _filter(output):
    _txt = t5_tokenizer.decode(output[2:])
    return _txt.replace('</s>', '')

results = list(map(_filter, outputs))
print(results[0])