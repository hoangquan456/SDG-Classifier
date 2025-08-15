from datasets import load_dataset, Dataset, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from huggingface_hub import HfFolder
import csv

filename = 'osdg-community-data-v2024-04-01.csv'

text = []
labels = []
with open(filename, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter = '\t')
    for i, row in enumerate(reader):
        if (i == 300000):
            break
        if (float(row['agreement']) < 0.65 
            or int(row['labels_negative']) + int(row['labels_positive']) <= 5 
            or int(row['labels_negative']) > int(row['labels_positive'])): 
            continue
        if (int(row['sdg']) == 16 and float(row['agreement']) < 0.8):
            continue
        text.append(row['text'])
        labels.append(int(row['sdg']))
train_dataset = Dataset.from_dict({'text': text, 'labels': labels})

train_valid_test = train_dataset.train_test_split(test_size=0.2, seed=42)
train_valid = train_valid_test["train"].train_test_split(test_size=0.125, seed=42)
split_dataset = DatasetDict({
    "train": train_valid["train"],
    "validation": train_valid["test"],
    "test": train_valid_test["test"]
})

# split_dataset = train_dataset.train_test_split(test_size=0.2)
# if "label" in split_dataset["train"].features.keys():
#     split_dataset =  split_dataset.rename_column("label", "labels")

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch['text'], padding= 'max_length', max_length = 768, truncation=True, return_tensors="pt")

tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])
print("tokenized dataset", tokenized_dataset)

num_labels = 17
 
# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # all_labels = np.unique(labels)

    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="macro")
    
    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

training_args = TrainingArguments(
    output_dir= "ModernBERT-SDG-classifier",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
        num_train_epochs=4,
    bf16=True, # bfloat16 training 
    optim="adamw_torch_fused", # improved optimizer 
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    use_mps_device=False,
    metric_for_best_model="f1",
    # push to hub parameters
    push_to_hub=False,
    hub_strategy="every_save",
    hub_token=HfFolder.get_token(),
    warmup_ratio = 0.1,
    weight_decay=0.01,  # Added for regularization
    max_grad_norm=1.0,  # Added for stability
    lr_scheduler_type="cosine",  # Explicitly set (optional: try "cosine")
)
 
# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("ModernBERT-SDG-classifier")

test_results = trainer.evaluate(tokenized_dataset["test"])
print(test_results)