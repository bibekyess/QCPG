import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt

class QualityControlPipeline:
    
    def __init__(self, qcpg_model_name, qp_model_name):
        self.qcpg = qcpg_model_name
        self.qp = qp_model_name
        self.pipe = pipeline('text2text-generation', model=self.qcpg)
        self.ranges = {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]}

    def get_reference_point(self, text):
        """
            Uses QP Model for inference
        """
        tokenizer = AutoTokenizer.from_pretrained(self.qp)
        model = AutoModelForSequenceClassification.from_pretrained(self.qp)

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        lexical_ = logits[0][0].item()
        syntactic_ = logits[0][1].item()
        semantic_ = logits[0][2].item()

        return lexical_, syntactic_, semantic_

    def __call__(self, text, lexical, syntactic, semantic, **kwargs):
        # assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
        #          f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']

        # Get reference points from pretrained QP Model
        lex_qp, syn_qp, sem_qp = self.get_reference_point(text)

        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control_qp = [int(5 * round(val * 100 / 5)) for val in [sem_qp, lex_qp, syn_qp]]
        # control = [control[i] + control_qp[i] for i in range(len(control))]
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)

# FIXME Change the pretrained qcpg and qp model name here
model = QualityControlPipeline(qcpg_model_name='jinujinu99/t5-ep3-parabk2', qp_model_name='ibm/qp-sentences')
input_sentence = 'We will do very good in the final project of this class.'

lex_range = np.arange(0,1.2,0.1)
responsiveness=[]

Q_0 = model.get_reference_point(input_sentence)
Q_0 = np.asarray(Q_0)
first = False
# Q_0 = Q_0/np.linalg.norm(Q_0)
for lex in lex_range:            
    result = model(input_sentence, lexical=lex, syntactic=0, semantic=0)
    sentence_paraphrased = result[0].get('generated_text')
    Q_n = model.get_reference_point(sentence_paraphrased)
    print(sentence_paraphrased)
    # Q_n = np.asarray(Q_n)
    # if first:
    #     Q_0 = Q_n
    #     r = 0
    #     first = False
    # else:
    #     # Q_n = Q_n/np.linalg.norm(Q_n)
    #     # print(np.linalg.norm(Q_0), np.linalg.norm(Q_n))
    #     r = np.linalg.norm(Q_0-Q_n)
    # print(lex, r, Q_n)
    # responsiveness.append(r)


# x_axis = list(lex_range)
# y_axis = responsiveness

# plt.plot(x_axis, y_axis)
# plt.title('Responsiveness of QCPG')
# plt.xlabel('Lexical offset')
# plt.ylabel('Responsiveness')
# plt.show()