import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

model_dir = "email_model"

tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

mlb = joblib.load(f"{model_dir}/mlb.pkl")

def predict_tags(text):

    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)

    preds = (probs > 0.2).int()

    if preds.sum() == 0:
        preds[0][torch.argmax(probs)] = 1

    tags = mlb.inverse_transform(preds.numpy())

    return tags[0]


print(predict_tags("I want pricing details"))