import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Load data
data = pd.read_csv("D:/resume screener/my_datasets/training_pairs.csv")

# Handle missing values
data['Resume'] = data['Resume'].fillna("")
data['Job_Desc'] = data['Job_Desc'].fillna("")

# Prepare labels if not numeric
if data['Label'].dtype != int and data['Label'].dtype != float:
    data['Label'] = data['Label'].astype(int)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    list(zip(data['Resume'], data['Job_Desc'])),
    data['Label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize with explicit string conversion
train_encodings = tokenizer(
    [str(resume) for resume, _ in train_texts],
    [str(job) for _, job in train_texts],
    truncation=True,
    padding="max_length",
    max_length=512
)

val_encodings = tokenizer(
    [str(resume) for resume, _ in val_texts],
    [str(job) for _, job in val_texts],
    truncation=True,
    padding="max_length",
    max_length=512
)

# Custom dataset
class ResumeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ResumeDataset(train_encodings, train_labels)
val_dataset = ResumeDataset(val_encodings, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(train_labels)))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train & evaluate
trainer.train()
trainer.evaluate()