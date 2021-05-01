# Romanian DistilBERT
This repository contains the DistilBERT version for Romanian. Teacher model used for distillation: [dumitrescustefan/bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1).

## Usage

Romanian DistilBERT is easily accesible via the HuggingFace interface.

```python
from transformers import AutoTokenizer, AutoModel

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("racai/distilbert-base-romanian-cased")
model = AutoModel.from_pretrained("racai/distilbert-base-romanian-cased")
# tokenize a test sentence
input_ids = tokenizer.encode("Aceasta este o propoziție de test.", add_special_tokens=True, return_tensors="pt")
# run the tokens trough the model
outputs = model(input_ids)
print(outputs)
```
## Model Size
Romanian DistilBERT is 35% smaller than the original Romanian BERT and approximately twice as fast.

| Model                          | Size (MB) | Params (Millions) |
|--------------------------------|:---------:|:----------------:| 
| bert-base-romanian-cased-v1    | 477.2 | 124.4 |
| distilbert-base-romanian-cased | 312.7 | 81.3 |

## Evaluation

We evaluated the Romanian DistilBERT in comparison with the original Romanian BERT on 5 tasks:

- **UPOS**: Universal Part of Speech
- **XPOS**: Extended Part of Speech
- **NER**: Named Entity Recognition
- **SAPN**: Sentiment Anlaysis - Positive vs Negative
- **SAR**: Sentiment Analysis - Rating
- **DI**: Dialect identification 

| Model                          | UPOS | XPOS | NER | SAPN | SAR | DI |
|--------------------------------|:----:|:----:|:---:|:----:|:---:|:--:|
| bert-base-romanian-cased-v1    | 98.00 | 96.46 | 85.88 | 98.07 | 79.61 | 95.58 |
| distilbert-base-romanian-cased | 97.97 | 97.08 | 83.35 | 98.40 | 83.01 | 96.31 |

More details can be found in the evaluation section of the repository.

## Citation

Coming soon.
