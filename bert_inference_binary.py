from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = "ModernBERT-SDG1-binary-classifier"
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sdgs(text):
    """
    Predict whether a text belongs to SDG1 or not.
    
    Args:
        text (str): Input text to classify
        
    Returns:
        dict: Dictionary containing the predicted class (1 for SDG1, 0 for not SDG1)
              and the confidence score
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding='max_length',
        max_length=768,
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to the appropriate device (CPU/GPU)
    device = "cuda"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)

    # Return results
    return [int(predicted_class)]
    return {
        "predicted_class": int(predicted_class),  # 1 for SDG1, 0 for not SDG1
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            "not_sdg1": float(probabilities[0]),
            "sdg1": float(probabilities[1])
        }
    }

print(predict_sdgs("""Many people feel insecure in their homes and communities. One billion girls and boys ages 2-17 worldwide experienced physical, sexual or psychological violence in the prior year, according to one study.94 Some 25 percent of children suffer physical abuse, and nearly 20 percent of girls are sexually abused at least once in their life.95 Elder abuse remains a hidden problem:96 10 percent of older adults were abused in the prior month.97 Homicide is also a major social concern. Physical security and freedom from the threat of violence were particular concerns among female respondents (box 2.7). For women, real or perceived physical and emotional violence is a major barrier to meeting their full human potential and feeling free to move about."""))