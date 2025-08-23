import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

MODEL_NAME = "bert-base-uncased"
DATA_PATH = "D:/resume screener/my_datasets/training_pairs.csv"
SAVE_PATH = "./resume_model_best"
NUM_EPOCHS = 4  
BATCH_SIZE = 16 
LEARNING_RATE = 2e-5

print("Loading and preparing data...")

data = pd.read_csv(DATA_PATH)
data['Resume'] = data['Resume'].fillna("")
data['Job_Desc'] = data['Job_Desc'].fillna("")
data['Label'] = data['Label'].astype(int)

texts = list(zip(data['Resume'], data['Job_Desc']))
labels = data['Label'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    [str(r) for r, j in train_texts], [str(j) for r, j in train_texts], 
    truncation=True, padding=True, max_length=512, return_tensors="pt"
)
val_encodings = tokenizer(
    [str(r) for r, j in val_texts], [str(j) for r, j in val_texts], 
    truncation=True, padding=True, max_length=512, return_tensors="pt"
)

class ResumeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ResumeDataset(train_encodings, train_labels)
val_dataset = ResumeDataset(val_encodings, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Calculating class weights...")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

weights_tensor = torch.tensor(class_weights, dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_tensor = weights_tensor.to(device)
print(f"Weights calculated to penalize misclassification: {weights_tensor.cpu().numpy()}")

print(f"Using device: {device}")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(train_dataloader) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    

    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        

        loss = loss_fn(outputs.logits, batch['labels'])
        
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    model.eval()
    val_preds, val_labels_list = [], []
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch['labels'])
            total_val_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            val_preds.extend(predictions.cpu().numpy())
            val_labels_list.extend(batch['labels'].cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(val_labels_list, val_preds)
    print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f"New best model found! Saving to {SAVE_PATH}")
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)

print("\nTraining complete!")
print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")