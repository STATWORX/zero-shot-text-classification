import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from utils import get_device
from datetime import datetime


def get_top_k_labels(obs, top_k: int = 3):
    """
    Extracts the top k labels from a transformers.pipeline object
    """
    labels = obs['labels']
    scores = obs['scores']
    # Returns the top_k indices with the highest scores, BUT note, they are not sorted!
    top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_k_labels = [labels[i] for i in top_k_idx]
    return top_k_labels


def prepare_pred_for_top_k(row):
    """
    Extracts and prepares scores and labels for @top k accuracy

    sklearn.metrics.top_k_accuracy_score expects for y_true an array-like of shape (n_samples, n_classes).
    This function will order the scores and labels in a unified way and return both as dict.
    """
    labels = row['pred_zs']['labels']
    scores = row['pred_zs']['scores']

    scores_ordered = [scores[i] for i in labels]
    labels_ordered = [labels[i] for i in labels]

    return {'labels': labels_ordered, 'scores': scores_ordered}


def predict_batch(batch, p, candidates: list, top_k: int = 3, template: str = "Das Thema ist {}"):
    """
    Applies a pipeline on batches from dataset object

    :param batch: batch of observations for prediction
    :param p: A hugging face transformers.pipeline object used for inference
    :param candidates: zero-shot candidates as list
    :param top_k: Number of top categories for accuracy @ top k
    :param template: hypothesis template
    :return: batch with predictions
    """
    pred = p(sequences=batch['text'],
             candidate_labels=candidates,
             hypothesis_template=template,
             multi_label=False)

    max_scores = [np.max(i['scores']) for i in pred]
    pred_labels = [i['labels'][np.argmax(i['scores'])] for i in pred]
    pred_labels_top_k = [get_top_k_labels(i) for i in pred]

    batch['pred_zs'] = pred
    batch['pred_zs_label'] = pred_labels
    batch['pred_zs_score'] = max_scores
    batch[f'pred_zs_label_top_{top_k}'] = pred_labels_top_k

    return batch


# Get the device (CPU, GPU, Apple M1/2 aka MPS)
# Ignoring MPS because this particular model contains some int64 ops that are not supported by the MPS backend yet :(
# see https://github.com/pytorch/pytorch/issues/80784
device = get_device(ignore_mps=True, cuda_as_int=True)

# Load the German version of the MultiLingual SUMmarization dataset from hugging face hub
# https://huggingface.co/datasets/mlsum
data = load_dataset('mlsum', 'de')
model = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
pipe = pipeline(task='zero-shot-classification', model=model, tokenizer=model, device=device)

# Topic candidates are the labels already present in the dataset
topic_candidates = np.unique(data['test']['topic'])
template_de = 'Das Thema ist {}'

# Take a random sample from test data
data['test'] = data['test'].shuffle(seed=42)
data['test'] = data['test'].select(range(1000))

# Apply the prediction function in batches to the dataset (test split)
map_kwargs = {'p': pipe, 'candidates': topic_candidates, 'template': template_de}
data['test'] = data['test'].map(predict_batch, batched=True, batch_size=16, fn_kwargs=map_kwargs)

# Save results to hard drive for later usage
data.save_to_disk('data/mlsum_predicted')

# Calculate accuracy
actual_label = data['test']['topic']
pred_zs_label = data['test']['pred_zs_label']
zs_accuracy = accuracy_score(y_true=actual_label, y_pred=pred_zs_label)
print(f'Zero-shot accuracy with "{template_de}": {zs_accuracy:.2%}')

# Calculate accuracy @top k
#pred_zs_scores = data['test'].map(prepare_pred_for_top_k, remove_columns=data['test'].column_names)
#pred_zs_scores = pred_zs_scores['scores']
#all_labels = np.sort(np.unique(data['train']['label']))
#zs_top_k_accuracy = top_k_accuracy_score(y_true=actual_label_int, y_score=pred_zs_scores, k=3, labels=all_labels)
#print(f'Zero-shot accuracy @top 3 with "{template_de}": {zs_top_k_accuracy:.2%}')

results = pd.DataFrame({'template': template_de,
                        'accuracy': zs_accuracy,
                        #'accuracy_top_3': zs_top_k_accuracy
                        },
                       index=[0])

# Prompt engineering / tuning with three different templates
templates = ['Im Artikel geht es um {}.',
             'Der Text ist über {}.',
             'In diesem geht es um {}.',
             'Thema: {}.']

# Run above process for various templates (tune prompt)
for t in templates:

    data_tmp = data.copy()
    map_kwargs = {'p': pipe, 'candidates': topic_candidates, 'template': t}
    data_tmp['test'] = data_tmp['test'].map(predict_batch, batched=True, batch_size=16, fn_kwargs=map_kwargs)

    pred_zs_label = data['test']['pred_zs_label']
    zs_accuracy = accuracy_score(y_true=actual_label, y_pred=pred_zs_label)
    print(f'Zero-shot accuracy with "{t}": {zs_accuracy:.2%}')

    #pred_zs_scores = data_tmp['test'].map(prepare_pred_for_top_k, remove_columns=data_tmp['test'].column_names)
    #pred_zs_scores = pred_zs_scores['scores']
    #zs_top_k_accuracy = top_k_accuracy_score(y_true=actual_label_int, y_score=pred_zs_scores, k=3, labels=all_labels)
    #print(f'Zero-shot accuracy @top 3 with "{t}": {zs_top_k_accuracy:.2%}')

    results = results.append(pd.DataFrame({'template': t,
                                           'accuracy': zs_accuracy,
                                           #'accuracy_top_3': zs_top_k_accuracy
                                           },
                                          index=[0]))

results['model'] = model
results['data'] = 'mlsum'
results = results.reset_index(drop=True)
results.to_pickle(f'results/{datetime.today().strftime("%Y-%m-%d")}_prompt_tuning.pickle')
