# Romanian DistilBERT
This repository contains the DistilBERT version for Romanian. Teacher model used for distillation: [dumitrescustefan/bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1).

## Usage
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
Romanian DistilBERT is 35% smaller than the original Romanian BERT.
| Model                          | Size (MB) | Params (Millions) |
|--------------------------------|:---------:|:----------------:| 
| bert-base-romanian-cased-v1    | 477.2 | 124.4 |
| distilbert-base-romanian-cased | 312.7 | 81.3 |
