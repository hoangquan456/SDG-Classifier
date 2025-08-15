import pickle
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from huggingface_hub import HfFolder

# Load the pickle files (arrays of texts)
with open('sdg1.pkl', 'rb') as f:
    sdg1_texts = pickle.load(f)
with open('not_sdg1_pro.pkl', 'rb') as f:
    not_sdg1_texts = pickle.load(f)

# Create lists for texts and labels
texts = sdg1_texts + not_sdg1_texts
labels = [1] * len(sdg1_texts) + [0] * len(not_sdg1_texts)

# Create DataFrame
df = pd.DataFrame({'text': texts, 'labels': labels})

# Create Dataset
dataset = Dataset.from_pandas(df)

# Split dataset into train, validation, and test
train_valid_test = dataset.train_test_split(test_size=0.2, seed=42)
train_valid = train_valid_test["train"].train_test_split(test_size=0.125, seed=42)
split_dataset = DatasetDict({
    "train": train_valid["train"],
    "validation": train_valid["test"],
    "test": train_valid_test["test"]
})

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', max_length=768, truncation=True, return_tensors="pt")

# Tokenize dataset
tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])

# Set up model for binary classification
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=num_labels,
    problem_type="single_label_classification"
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1 = f1_score(labels, predictions, average="binary")
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    
    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

# Training arguments optimized for binary classification
training_args = TrainingArguments(
    output_dir="ModernBERT-SDG1-binary-classifier",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=4,
    bf16=True,
    optim="adamw_torch_fused",
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    use_mps_device=False,
    metric_for_best_model="f1",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_token=HfFolder.get_token(),
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("ModernBERT-SDG1-binary-classifier")

# Evaluate on test set
test_results = trainer.evaluate(tokenized_dataset["test"])
print(test_results)