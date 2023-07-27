import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PATH = 'khvatov/ru_toxicity_detector'
tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH)

model.to(torch.device("cpu"))


def get_toxicity_probs(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.nn.functional.softmax(model(**inputs).logits, dim=1).cpu().numpy()
    return proba[0]


TEXT = "Я тебя люблю"
print(f'text = {TEXT}, probs={get_toxicity_probs(TEXT)}')