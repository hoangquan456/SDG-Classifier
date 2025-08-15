from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "./ModernBERT-SDG-classifier/"
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModelForSequenceClassification.from_pretrained(model_id).to("cuda")

def predict_sdg(text):
    inputs = tokenizer(text, return_tensors = "pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return predicted_class
