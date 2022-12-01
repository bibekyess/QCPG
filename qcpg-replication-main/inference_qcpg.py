import torch
from transformers import T5Tokenizer, MT5Config, MT5ForConditionalGeneration

T5_PATH = 'lcw99/t5-base-korean-paraphrase' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
DEVICE = 'cpu'

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = MT5Config.from_pretrained(T5_PATH)
t5_mlm = MT5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Input text
# COND_SEMANTIC_SIM_80 COND_LEXICAL_DIV_30 COND_SYNTACTIC_DIV_50Is this going to work or what are we doing here?
# Very interesting that putting _ infront of number gives error..
text = '이것이 작동할까요, 아니면 여기서 무엇을 하는 걸까요?'

encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

outputs = t5_mlm.generate(input_ids=input_ids)
print(outputs)

# _0_index = text.index('<extra_id_0>')
# _result_prefix = text[:_0_index]
# _result_suffix = text[_0_index+12:]  # 12 is the length of <extra_id_0>

def _filter(output):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[2:])
    return _txt 
results = list(map(_filter, outputs))
print(results)