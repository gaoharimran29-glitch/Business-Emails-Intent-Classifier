import pandas as pd
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"dataset/email_intent_dataset_fixed.csv")

df["tags"] = df["tags"].apply(lambda x: [t.strip() for t in x.split(",")])

texts = df["email_text"].astype(str).tolist()

# Better prompt
texts = ["Email intent classification. Identify the business intent: " + t for t in texts]

# -----------------------------
# Encode labels
# -----------------------------
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df["tags"])

print("Tag classes:", mlb.classes_)

# -----------------------------
# Train test split
# -----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding="max_length",
    max_length=128
)

# -----------------------------
# Dataset
# -----------------------------
class EmailDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

# -----------------------------
# Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)

# -----------------------------
# Class weights
# -----------------------------
class_counts = labels.sum(axis=0)
total_samples = len(labels)

class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# -----------------------------
# Custom Trainer
# -----------------------------
class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=class_weights.to(model.device)
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# -----------------------------
# Metrics (threshold based)
# -----------------------------
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits))

    threshold = 0.2

    preds = (probs > threshold).int().numpy()

    return {
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0)
    }

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    seed=42
)

# -----------------------------
# Trainer
# -----------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Evaluation
# -----------------------------
results = trainer.evaluate()

print("\nEvaluation Results:")
print(results)

# -----------------------------
# Save metrics
# -----------------------------
with open("metrics.txt", "w") as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n")

print("Metrics saved to metrics.txt")

# -----------------------------
# Save model
# -----------------------------
model_dir = "models"

trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

joblib.dump(mlb, f"{model_dir}/mlb.pkl")

print("\nModel saved to:", model_dir)

# -----------------------------
# Plot training loss
# -----------------------------
logs = trainer.state.log_history

losses = [x["loss"] for x in logs if "loss" in x]

plt.figure()
plt.plot(losses)

plt.title("Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")

plt.savefig("training_loss_tags_classifier.png")

print("Training loss chart saved")

plt.close()

# -----------------------------
# Prediction function
# -----------------------------
def predict_tags(text):

    model.eval()

    text = "Email intent classification. Identify the business intent: " + text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)

    threshold = 0.2

    preds = (probs > threshold).int()

    preds = preds.cpu().numpy()

    tags = mlb.inverse_transform(preds)

    return tags[0]

# -----------------------------
# Test predictions
# -----------------------------
test_emails = [

    "i want to schedule a demo of your crm",

    "your platform crashed while exporting contacts",

    "how do i integrate crm with slack",

    "please remove me from your mailing list",

    "can you share pricing details for enterprise plan"

]

print("\nSample Predictions:\n")

for email in test_emails:

    tags = predict_tags(email)

    print("Email:", email)
    print("Predicted Tags:", tags)
    print()