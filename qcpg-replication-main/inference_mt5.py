from transformers import pipeline

class QualityControlPipeline:
    
    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'bibekyess/qcpg-parabk2-mt5')
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
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        # text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)

model = QualityControlPipeline('sentences')

print(model('Is this going to work or what are we doing here?', lexical=0.3, syntactic=0.5, semantic=0.8)
)