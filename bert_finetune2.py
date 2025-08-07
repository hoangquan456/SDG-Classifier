from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
import numpy as np
from huggingface_hub import HfFolder
import csv

# dataset_id = "argilla/synthetic-domain-text-classification"

# train_dataset = load_dataset(dataset_id, split = 'train')

filename = 'osdg-community-data-v2024-04-01.csv'

text = []
labels = []
with open(filename, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter = '\t')
    for i, row in enumerate(reader):
        if (float(row['agreement']) < 0.65 
            or int(row['labels_negative']) + int(row['labels_positive']) <= 5 
            or int(row['labels_negative']) > int(row['labels_positive'])): 
            continue
        text.append(row['text'])
        labels.append(int(row['sdg']))
        if (i == 6000): 
            break
train_dataset = Dataset.from_dict({'text': text, 'labels': labels})
# class_names = list(range(1, 18))
# train_dataset = train_dataset.cast_column(
#     "labels",
#     ClassLabel(names=class_names)
# )

# # tmp.features['labels'] = ClassLabel(names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'])
# print(tmp.features['labels'])

# exit(0)

# print("train dataset", train_dataset)
# print("new dataset", tmp)

split_dataset = train_dataset.train_test_split(test_size=0.1)
if "label" in split_dataset["train"].features.keys():
    split_dataset =  split_dataset.rename_column("label", "labels")

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch['text'], padding= 'max_length', max_length = 768, truncation=True, return_tensors="pt")

tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])
print("tokenized dataset", tokenized_dataset)

num_labels = 17
all_labels = []
 
# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # all_labels = np.unique(labels)

    score = f1_score(
            labels, predictions, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}

training_args = TrainingArguments(
    output_dir= "ModernBERT-domain-classifier",
    per_device_train_batch_size=20,
    per_device_eval_batch_size=40,
    learning_rate=5e-5,
        num_train_epochs=5,
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
    warmup_ratio = 0.1
)
 
# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()