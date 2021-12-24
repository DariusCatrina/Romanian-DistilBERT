# Romanian DistilBERT
This repository contains the DistilBERT models for Romanian described in [this paper](https://arxiv.org/abs/2112.12650):
- **distilbert-base-romanian-uncased** (Distil-RoBERT-base) - teacher(s): [dumitrescustefan/bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1).
- **distilbert-base-romanian-cased** (Distil-BERT-base-ro) - teacher(s): [readerbench/RoBERT-base](https://huggingface.co/readerbench/RoBERT-base).
- **distilbert-multi-base-romanian-cased** (DistilMulti-BERT-base-ro) - teachers(s): [dumitrescustefan/bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1) and [readerbench/RoBERT-base](https://huggingface.co/readerbench/RoBERT-base).

## Usage

Romanian DistilBERTs are easily accesible via the HuggingFace interface. This is an example for `distilbert-base-romanian-cased`:

```python
from transformers import AutoTokenizer, AutoModel

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("racai/distilbert-base-romanian-cased")
model = AutoModel.from_pretrained("racai/distilbert-base-romanian-cased")

# tokenize a test sentence
input_ids = tokenizer.encode("Aceasta este o propoziție de test.", add_special_tokens=True, return_tensors="pt")
# run the tokens trough the model
outputs = model(input_ids)
```
## Model Size
The models are roughly 35% smaller than their original teachers and approximately twice as fast.

| Model                          | Size (MB) | Params (Millions) |
|--------------------------------|:---------:|:----------------:| 
| RoBERT-base    | 441 | 114 |
| bert-base-romanian-cased-v1    | 477 | 124 |
| distilbert-base-romanian-uncased | 282 | 72 |
| distilbert-base-romanian-cased | 312 | 81 |
| distilbert-multi-base-romanian-cased | 312 | 81 |

## Evaluation

We evaluated the models in comparison with Romanian BERTs on 5 tasks:

- **UPOS**: Universal Part of Speech (F1-macro)
- **XPOS**: Extended Part of Speech (F1-macro)
- **NER**: Named Entity Recognition (F1-macro)
- **SAPN**: Sentiment Anlaysis - Positive vs Negative (Accuracy)
- **SAR**: Sentiment Analysis - Rating (F1-macro)
- **DI**: Dialect identification  (F1-macro)
- **STS**: Semantic Textual Similarity (Pearson)

They maintain most of the performance of their teachers:

| Model                          | UPOS | XPOS | NER | SAPN | SAR | DI | STS |
|--------------------------------|:----:|:----:|:---:|:----:|:---:|:--:|:---:|
| RoBERT-base | 98.02 | 97.15 | 85.14 | 98.30 | 79.40 | 96.07 | 81.18 |
| bert-base-romanian-cased-v1    | 98.00 | 96.46 | 85.88 | 98.07 | 79.61 | 95.58 | 79.11 |
| distilbert-base-romanian-uncased | 97.12 | 95.79 | 83.11 | 98.01 | 79.58 | 96.11 | 79.80 |
| distilbert-base-romanian-cased | 97.97 | 97.08 | 83.35 | 98.20 | 80.51 | 96.31 | 80.57 |
| distilbert-multi-base-romanian-cased | 98.07 | 96.83 | 83.22 | 98.11 | 79.77 | 96.18 | 80.66 |

## Credits

Please consider citing the following [paper](https://arxiv.org/abs/2112.12650) as a thank you to the authors of Romanian DistilBERTs: 
```
Andrei-Marius Avram, Darius Catrina, Dumitru-Clementin Cercel, Mihai Dascălu, Traian Rebedea, Vasile Păiş and Dan Tufiş. "Distilling the Knowledge of Romanian BERTs Using Multiple Teachers" arXiv preprint arXiv:2112.12650 (2021).
```
or in .bibtex format:
```
@article{avram2021distilling,
  title={Distilling the Knowledge of Romanian BERTs Using Multiple Teachers},
  author={Andrei-Marius Avram and Darius Catrina and Dumitru-Clementin Cercel and Mihai Dascălu and Traian Rebedea and Vasile Păiş and Dan Tufiş},
  journal={ArXiv},
  year={2021},
  volume={abs/2112.12650}
}
```
