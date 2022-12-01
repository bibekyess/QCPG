import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("ibm/qp-sentences")

model = AutoModelForSequenceClassification.from_pretrained("ibm/qp-sentences")

inputs = tokenizer("Is this going to work or what are we doing here?", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

lexical = logits[0][0].item()
syntactic = logits[0][1].item()
semantic = logits[0][2].item()

print(lexical, syntactic, semantic)
class QualityControlPipeline:
    
    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def __call__(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
                 f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']

        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        print(control)
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        print(control)
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        print(control)
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        print(control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        print(text, **kwargs)
        print(self.pipe(text, **kwargs))
        return self.pipe(text, **kwargs)

model = QualityControlPipeline('sentences')
model('Is this going to work or what are we doing here?', lexical=0.3, syntactic=0.5, semantic=0.8)
