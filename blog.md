# Zero-Shot Text Classification

Text classification is one of the most common applications for natural language processing (NLP). It is the task of 
assigning a set of predefined categories to a text snippet. Depending on the type of problem, the text snippet could be 
a sentence, a paragraph, or even a whole document. There are many potential real-world applications for text classification, 
but among the most frequent ones are sentiment analysis, topic, intent, spam, and hate speech classification.

The standard approach to text classification is training a classifier in a supervised regime. To do so, one needs pairs 
of text and associated categories (aka labels) from the domain of interest as training data. Then, any classifier (e.g., a neural network) 
can learn a mapping function from the text to the most likely category. While this approach can work quite well for many 
settings, its feasibility highly depends on the availability of those hand-labeled training pairs. 

Though pre-trained language models like BERT can reduce the amount of data needed, it does not make it 
obsolete altogether. For real-world applications, data availability remains therefore the biggest hurdle.


## Zero-Shot Learning
Though there are various definitions of [zero-shot learning][1], it can broadly be speaking defined as a regime in which a 
model solves a task it was not explicitly trained on before. 
It is important to understand, that a “task” can be defined in a broader and narrower sense: For example, the [authors of 
GPT-2 showed that a model trained on language generation can be applied to entirely new downstream tasks like machine 
translation][2]. At the same time, it can also mean being able to recognize previously unseen categories in images as 
shown in the OpenAI [CLIP paper][3].  

But what all these approaches have in common is the idea of extrapolation of learned concepts beyond the training regime. 
A powerful concept, because it disentangles the solvability of a task from the availability of (labeled) training data.

## Zero-Shot Learning for Text Classification
Solving text classification tasks with zero-shot learning can serve as a good example of how to apply the extrapolation 
of learned concepts beyond the training regime. One way to do this is using natural language inference (NLI) as 
proposed by [Yin et al. (2019)][4]. There are other approaches as well like the calculation of distances between text 
embeddings or formulating the problem as a cloze task1.

In NLI the task is to determine whether a hypothesis is true (entailment), false (contradiction), or undetermined 
(neutral) given a [premise][5]. A typical NLI dataset consists of sentence pairs with associated labels in the following form:

![](https://joeddav.github.io/blog/images/zsl/nli-examples.png)
Examples from http://nlpprogress.com/english/natural_language_inference.html

Yin et al. (2019) proposed to use large language models like BERT trained on NLI datasets and exploit their language 
understanding capabilities for zero-shot text classification. This can be done by taking the text of interest as the 
premise and formulating one hypothesis for each potential category by using a so-called hypothesis template. 
Then, we let the NLI model predict whether the premise entails the hypothesis. Finally, the predicted probability 
of entailment can be interpreted as the probability of the label.

## Zero-Shot Text Classification with Hugging Face 🤗

Let’s explore the above-formulated idea in more in detail using the excellent Hugging Face implementation for zero-shot 
text classification.

We are interested in classifying the sentence below into pre-defined topics:

```python
topics = ['Web', 'Panorama', 'International', 'Wirtschaft', 'Sport', 'Inland', 'Etat', 'Wissenschaft', 'Kultur']
test_txt = 'Eintracht Frankfurt gewinnt die Europa League nach 6:5-Erfolg im Elfmeterschießen gegen die Glasgow Rangers'
```

As written above, we need a language model that was pre-trained on an NLI task. The default model for zero-shot text 
classification in 🤗 is `bart-large-mnli`. [BART is a transformer encoder-decoder for sequence-2-sequence modeling with a 
bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder][6]. The `mnli` suffix means that BART was then 
fine-tuned on the [MultiNLI dataset][7].

But since we are using German sentences and BART is English-only, we need to replace the default model with a custom one. 
Thanks to the 🤗 model hub, finding a suitable candidate is quite easy. In our case, 
`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` is such a candidate. Let’s decrypt the name shortly for a better 
understanding: it is a multilanguage version of DeBERTa-v3-base (which is itself an improved version of [BERT/RoBERTa][8]) 
that was then fine-tuned on two cross-lingual NLI datasets ([XNLI][9] and [multilingual-NLI-26lang][10]).

With the correct task and the correct model, we can now instantiate the pipeline:

```python
from transformers import pipeline
model = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
pipe = pipeline(task='zero-shot-classification', model=model, tokenizer=model)
```

Next, we call the pipeline to predict the most likely category of our text given the candidates. But as a final step, 
we need to replace the default hypothesis template as well. This is necessary since the default is again in English. 
We, therefore, define the template as `Das Thema is {}`. Note that, `{}` is a placeholder for the previously defined 
topic candidates. You can define any template you like as long as it contains a placeholder for the candidates:

```python
template_de = 'Das Thema ist {}'
prediction = pipe(test_txt, topics, hypothesis_template=template_de)
```

Finally, we can assess the prediction from the pipeline. The code below will output the three most likely topics 
together with their predicted probabilities:

```python
print(f'Zero-shot prediction for: \n {prediction["sequence"]}')
top_3 = zip(prediction['labels'][0:3], prediction['scores'][0:3])
for label, score in top_3:
    print(f'{label} - {score:.2%}')
```

```
Zero-shot prediction for: 
 Eintracht Frankfurt gewinnt die Europa League nach 6:5-Erfolg im Elfmeterschießen gegen die Glasgow Rangers
Sport - 77.41%
International - 15.69%
Inland - 5.29%
```

As one can see, the zero-shot model produces a reasonable result with “Sport” being the most likely topic followed by 
“International” and “Inland”.

Below are a few more examples from other categories. Like before, the results are overall quite reasonable, only for 
the second text, the model puts an unexpectable low probability on “Kultur”.

```python
further_examples = ['Verbraucher halten sich wegen steigender Zinsen und Inflation beim Immobilienkauf zurück',
                    '„Die bitteren Tränen der Petra von Kant“ von 1972 geschlechtsumgewandelt und neu verfilmt',
                    'Eine 541 Millionen Jahre alte fossile Alge weist erstaunliche Ähnlichkeit zu noch heute existierenden Vertretern auf']

for txt in further_examples:
    prediction = pipe(txt, topics, hypothesis_template=template_de)
    print(f'Zero-shot prediction for: \n {prediction["sequence"]}')
    top_3 = zip(prediction['labels'][0:3], prediction['scores'][0:3])
    for label, score in top_3:
        print(f'{label} - {score:.2%}')
```

```
Zero-shot prediction for: 
 Verbraucher halten sich wegen steigender Zinsen und Inflation beim Immobilienkauf zurück
Wirtschaft - 96.11%
Inland - 1.69%
Panorama - 0.70%

Zero-shot prediction for: 
 „Die bitteren Tränen der Petra von Kant“ von 1972 geschlechtsumgewandelt und neu verfilmt
International - 50.95%
Inland - 16.40%
Kultur - 7.76%

Zero-shot prediction for: 
 Eine 541 Millionen Jahre alte fossile Alge weist erstaunliche Ähnlichkeit zu noch heute existierenden Vertretern auf
Wissenschaft - 67.52%
Web - 8.14%
Inland - 6.91%
```

The entire code can be found on GitHub as well. Besides the examples from above, you will find there also applications 
of zero-shot text classifications on two labeled datasets including an evaluation of the accuracy. In addition, I added 
some prompt-tuning by playing around with the hypothesis template.

## Concluding Thoughts
Zero-shot text classification offers a suitable approach when either training data is limited (or even non-existing) 
or as an easy-to-implement benchmark for more sophisticated methods. While explicit approaches, like fine-tuning large 
pre-trained models, certainly still outperform implicit approaches, like zero-shot learning, their universal 
applicability make them very appealing.

In addition, we should expect zero-shot learning, in general, to become more important over the next few years. 
This is because the way we will use models to solve tasks will evolve with the increasing importance of large 
pre-trained models. Therefore, I advocate that already today zero-shot techniques should be considered part of every 
modern data scientist’s toolbox. 

[1]: https://joeddav.github.io/blog/2020/05/29/ZSL.html
[2]: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[3]: https://arxiv.org/pdf/2103.00020.pdf
[4]: https://arxiv.org/pdf/1909.00161.pdf 
[5]: http://nlpprogress.com/english/natural_language_inference.html
[6]: https://arxiv.org/pdf/1910.13461.pdf
[7]: https://huggingface.co/datasets/multi_nli
[8]: https://arxiv.org/pdf/2006.03654.pdf
[9]: https://huggingface.co/datasets/xnli
[10]: https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7
