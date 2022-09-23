from transformers import pipeline
from utils import get_device


# Get the device (CPU, GPU, Apple M1/2 aka MPS)
# Ignoring MPS because this particular model contains some int64 ops that are not supported by the MPS backend yet :(
# see https://github.com/pytorch/pytorch/issues/80784
device = get_device(ignore_mps=True, cuda_as_int=True)

# Define a model from hugging face hub: https://huggingface.co/models
model = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

# Example sentence to be classified
test_txt = 'Eintracht Frankfurt gewinnt die Europa League nach 6:5-Erfolg im Elfmeterschießen gegen die Glasgow Rangers'

# Define a german hypothesis template and the potential candidates for entailment/contradiction
template_de = 'Das Thema ist {}'
topics = ['Web', 'Panorama', 'International', 'Wirtschaft', 'Sport', 'Inland', 'Etat', 'Wissenschaft', 'Kultur']

# Pipeline abstraction from hugging face
pipe = pipeline(task='zero-shot-classification', model=model, tokenizer=model, device=device)

# Run pipeline with a test case
prediction = pipe(test_txt, topics, hypothesis_template=template_de)

# Top 3 topics as predicted in zero-shot regime
print(f'Zero-shot prediction for: \n {prediction["sequence"]}')
top_3 = zip(prediction['labels'][0:3], prediction['scores'][0:3])
for label, score in top_3:
    print(f'{label} - {score:.2%}')


# Some more examples?
further_examples = ['Verbraucher halten sich wegen steigender Zinsen und Inflation beim Immobilienkauf zurück',
                    '„Die bitteren Tränen der Petra von Kant“ von 1972 geschlechtsumgewandelt und neu verfilmt',
                    'Eine 541 Millionen Jahre alte fossile Alge weist erstaunliche Ähnlichkeit zu noch heute existierenden Vertretern auf']

for txt in further_examples:
    prediction = pipe(txt, topics, hypothesis_template=template_de)
    print(f'Zero-shot prediction for: \n {prediction["sequence"]}')
    top_3 = zip(prediction['labels'][0:3], prediction['scores'][0:3])
    for label, score in top_3:
        print(f'{label} - {score:.2%}')
