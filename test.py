import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "D:/resume screener/resume_model" 
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device) 
model.eval()     

test_data = pd.read_csv("D:/resume screener/my_datasets/training_pairs.csv")

test_data['Resume'] = test_data['Resume'].fillna("")
test_data['Job_Desc'] = test_data['Job_Desc'].fillna("")
y_true = test_data['Label'].tolist()
test_texts = list(zip(test_data['Resume'], test_data['Job_Desc']))

all_predictions = []

with torch.no_grad():
    for i in tqdm(range(0, len(test_texts), 8)):
        batch_texts = test_texts[i:i+8]
        
        inputs = tokenizer(
            [str(resume) for resume, _ in batch_texts],
            [str(job) for _, job in batch_texts],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt" 
        )
        
        inputs = {key: val.to(device) for key, val in inputs.items()}

        outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())

y_pred = np.array(all_predictions)

print("\n--- Classification Report ---")
target_names = ['Not a Fit', 'Fit']
print(classification_report(y_true, y_pred, target_names=target_names))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()