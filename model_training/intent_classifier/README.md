# Email Intent Classification Module
### CRM AI Side Task – Part 1: Intent Classification

---

## Objective
Classify incoming CRM emails into one of 5 intent categories using Machine Learning.

**Intents Supported:**
| Intent | Description |
|---|---|
| Pricing Inquiry | Customer asking about cost, plans, billing |
| Demo Request | Customer wants to see the product live |
| Complaint | Customer reporting a problem or expressing dissatisfaction |
| General Question | Customer asking how to use features |
| Not Interested | Customer declining or opting out |

---

## Approach

**Method:** TF-IDF Vectorisation + Logistic Regression

| Step | Detail |
|---|---|
| Text Cleaning | Lowercase + strip whitespace |
| Vectorisation | TF-IDF with unigrams + bigrams, top 5000 features |
| Classifier | Logistic Regression (C=5.0, max_iter=1000) |
| Train/Test Split | 80% train / 20% test, stratified |

**Why TF-IDF + Logistic Regression?**
- Fast to train on small datasets
- Highly interpretable
- Strong baseline for text classification
- No GPU required

---

## Project Structure

```
email_intent/
│
├── dataset.py              # Synthetic dataset (100 samples, 5 classes)
├── train_and_predict.py    # Training, evaluation, and prediction
│
├── data/
│   └── email_intent_dataset.csv   # Saved dataset (generated on run)
│
├── model/
│   ├── vectorizer.pkl      # Trained TF-IDF vectorizer
│   └── intent_model.pkl    # Trained Logistic Regression model
│
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy
```

### 2. Train the model & see predictions
```bash
python train_and_predict.py
```

This will:
- Generate and save the dataset to `data/`
- Train the model and print evaluation metrics
- Save the model to `model/`
- Print 10 example predictions

### 3. Use the model in your own code
```python
from train_and_predict import load_model, predict_intent

vectorizer, model = load_model()

result = predict_intent("Can you send me the pricing for 50 users?", vectorizer, model)
print(result["intent"])         # Pricing Inquiry
print(result["confidence"])     # {'Complaint': 1.2, 'Demo Request': 5.3, ...}
```

---

## Model Performance

**Test-set Accuracy: 75%** (on 20 held-out samples from 100 total)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Complaint | 1.00 | 0.50 | 0.67 |
| Demo Request | 1.00 | 0.50 | 0.67 |
| General Question | 0.50 | 0.75 | 0.60 |
| Not Interested | 0.80 | 1.00 | 0.89 |
| Pricing Inquiry | 0.80 | 1.00 | 0.89 |

> Note: Accuracy can be improved significantly with a larger dataset (500+ samples) or by upgrading to DistilBERT.

---

## Example Predictions

| Email | Predicted Intent | Confidence |
|---|---|---|
| "Can you send pricing details for enterprise plan?" | Pricing Inquiry | 72.66% |
| "I would like a demo of your CRM platform next week." | Demo Request | 49.76% |
| "Your software keeps crashing every time I open it!" | Complaint | 53.42% |
| "How do I integrate your CRM with Zapier?" | General Question | 59.97% |
| "We are not interested. Please remove us from your list." | Not Interested | 75.49% |
| "I want to see a demo and also know the pricing." | Demo Request | 49.78% |
| "What is the refund policy if we cancel early?" | Pricing Inquiry | 61.41% |
| "The billing page is broken – I can't update my card." | Complaint | 34.51% |
| "Can I get a free trial before committing to a plan?" | Pricing Inquiry | 50.67% |
| "We have selected a different vendor. Thanks anyway." | Not Interested | 59.38% |

---

## Dataset
- **Type:** Synthetic (manually crafted, no real company emails used)
- **Size:** 100 samples (20 per class)
- **Format:** CSV with columns `email_text`, `intent`
- **Location:** `data/email_intent_dataset.csv`

---

## Future Improvements
- Use DistilBERT for higher accuracy on ambiguous emails
- Expand dataset to 500+ samples
- Add confidence threshold to flag "uncertain" predictions for human review
