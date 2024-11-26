import os
import json
import boto3
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/tmp"

# S3 bucket and key prefix
S3_BUCKET = "emotion-detection-model"
MODEL_KEY_PREFIX = "emotion_model/"
LOCAL_MODEL_DIR = "/tmp/emotion_model"

# Function to download the model files from S3
def download_model():
    try:
        if not os.path.exists(LOCAL_MODEL_DIR):
            os.makedirs(LOCAL_MODEL_DIR)
            print(f"Created directory {LOCAL_MODEL_DIR}")
        
        s3 = boto3.client("s3")
        files = [
            "config.json", 
            "model.safetensors", 
            "special_tokens_map.json", 
            "tokenizer_config.json", 
            "tokenizer.json", 
            "vocab.txt"
        ]

        for file in files:
            local_path = os.path.join(LOCAL_MODEL_DIR, file)
            print(f"Downloading {file} to {local_path}")
            s3.download_file(S3_BUCKET, f"{MODEL_KEY_PREFIX}{file}", local_path)
            print(f"Downloaded {file}")

        print(f"Model files downloaded successfully to {LOCAL_MODEL_DIR}")
    except Exception as e:
        print(f"Error downloading model files: {e}")
        raise

# Cache model on first invocation
download_model()

# Load model and tokenizer
try:
    print(f"Loading model from {LOCAL_MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    print("Model and tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

# Lambda handler function
def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event.get("body", "{}"))
        input_text = body.get("text", "")

        if not input_text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No text provided"}),
            }

        # Tokenize input and get predictions
        print(f"Processing input text: {input_text}")
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Map predicted class to emotion
        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        emotion = labels[predicted_class]
        print(f"Predicted emotion: {emotion}")

        return {
            "statusCode": 200,
            "body": json.dumps({"emotion": emotion}),
            "headers": {"Content-Type": "application/json"},
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
