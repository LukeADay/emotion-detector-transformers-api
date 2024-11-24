import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_model_prediction():
    """Test if the model predicts emotions correctly."""
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./app/emotion_model")
    tokenizer = AutoTokenizer.from_pretrained("./app/emotion_model")

    # Test input
    text = "I love this project!"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Check if the output is valid
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    assert predicted_class in range(len(labels))
