# Zero Shot Text Classification

Some examples on how to apply zero-shot on text classification using the 🤗 transformers library.   

`zero_shot_example.py` showcases some examples, while `zero_shot_gnad10.py` and `zero_shot_mlsum.py` applies the idea 
on to German topic classification datasets. Performance evaluation can be found below. 

## Zero Shot Accuracy on Test Data

| Dataset                                         | Accuracy @top 1 | Accuracy @top 3 |
|-------------------------------------------------|-----------------|-----------------|
| Gnad10 (https://huggingface.co/datasets/gnad10) | 42 - 44%        | 47 - 50%        |
| MLSUM (https://huggingface.co/datasets/mlsum)   | 58%             | 82 - 85%        |

## Blog Post

See the full blog post on www.statworx.com or in [this file](blog.md). 

