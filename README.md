# Emotion Detection API

A REST API for detecting emotions in text, using a fine-tuned transformer model (`DistilBERT`) trained on the Emotion Dataset.

## Features
- Predicts one of six emotions: **anger**, **disgust**, **fear**, **joy**, **sadness**, **surprise**.
- Powered by Hugging Face's `transformers` library and Flask for deployment.
- Dockerized for easy deployment.

---

## Directory Structure

emotion-detection/
├── app/
│   ├── main.py            # Flask application
│   ├── model/             # Directory for the fine-tuned model
│   └── templates/         # Optional HTML templates for UI (if needed)
├── fine_tune.py           # Script to fine-tune the model
├── Dockerfile             # Docker configuration for containerizing the app
├── requirements.txt       # Dependencies for the Flask app and model
├── README.md              # Project description and setup instructions
└── .gitignore             # Files to exclude from version control

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

### 2. Create a Python virtual environment
Use `pyenv` or `venv` to create and activate a virtual environment:

```bash
pyenv virtualenv 3.10.10 emotion-env
pyenv activate emotion-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Fine-Tune the Model
Run the `fine_tune.py` script to train and save the model:

```bash
python fine_tune.py
```

### 2. Start the Flask App

```bash
cd app
python main.py
```

The API will be available at `http://127.0.0.1:8000`.

### 3. Test the API

Make a POST request with sample text:

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"text": "I am so excited for this project!"}' \
http://127.0.0.1:8000/predict
```

**Expected Response:**
```json
{"emotion": "joy"}
```

---

## Deployment

### Using Docker

1. Build the Docker image:
```bash
docker build -t emotion-detection .
```

2. Run the container
```bash
docker run -p 8000:8000 emotion-detection
```

---

## Technologies Used

* **Hugging Face Transformers**: For the fine-tuned `DistilBERT` model.
* Flask: For building the REST API.
* Docker: For containerized deployment.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

