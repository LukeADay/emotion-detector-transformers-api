from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt

# Load the test dataset
dataset = load_dataset("emotion")
test_dataset = dataset["test"]

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./app/emotion_model")
tokenizer = AutoTokenizer.from_pretrained("./app/emotion_model")

# Define the metric
accuracy_metric = load("accuracy")

# Map labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Store predictions and true labels
y_true = []
y_pred = []

# Evaluate on test set
for item in test_dataset:
    text = item["text"]
    true_label = item["label"]

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Collect predictions and true labels
    y_true.append(true_label)
    y_pred.append(predicted_class)

    # Update the accuracy metric
    accuracy_metric.add_batch(predictions=[predicted_class], references=[true_label])

# Compute the final accuracy
accuracy = accuracy_metric.compute()
print(f"Accuracy on test set: {accuracy['accuracy']:.4f}")

# Calculate precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Generate and display the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
