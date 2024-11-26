# Emotion Detection API

[![Build and Deploy to AWS Lambda](https://github.com/LukeADay/emotion-detector-transformers-api/actions/workflows/deploy.yaml/badge.svg)](https://github.com/LukeADay/emotion-detector-transformers-api/actions/workflows/deploy.yaml/badge.svg)

A REST API for detecting emotions in text, using a fine-tuned transformer model (`DistilBERT`) trained on the Emotion Dataset. Deployed via AWS Lambda and API Gateway for serverless operation, with CI/CD managed through GitHub Actions and AWS CodeBuild.

### Quick Start: Querying the API

To get started quickly, you can query the API with the following `curl` command:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so happy today!"}' \
  https://9fuctupzoj.execute-api.eu-west-2.amazonaws.com/prod/predict

```

---

## Features
- Predicts one of six emotions: **anger**, **disgust**, **fear**, **joy**, **sadness**, **surprise**.
- Powered by Hugging Face's `transformers` library for deployment.
- Serverless deployment with AWS Lambda and API Gateway.
- Fully automated CI/CD pipeline using GitHub Actions.
- Dockerised for both development and deployment.

---

## Pipeline Overview

1. **Model Training**:
- Fine-tune the `DistilBERT` model on the Emotion Dataset using `fine_tune.py`.
- The fine-tuned model is saved in the `app/emotion_model` directory and uploaded to an `S3` bucket.

2. **CodeBuild & Docker**:
- `AWS CodeBuild` uses a `Dockerfile` to build a `Lambda`-compatible container.
- The container is used to deploy the Lambda function, ensuring compatibility and portability.

3. **Deployment**:

- The Lambda function (`lambda_function.py`) loads the model from `S3`, processes requests, and predicts emotions.
- `API Gateway` routes incoming requests to the Lambda function.

4. CI/CD Pipeline:
- GitHub Actions (`deploy.yml`) automates:
    + Testing with `pytest`.
    + Triggering AWS CodeBuild to package the Lambda function.
    + Deploying the Lambda function and updating API Gateway.

5. API Gateway:
- API Gateway provides a POST endpoint `/predict` for consuming the service.

---

## Directory Structure

```
|── .github/workflows/
│   └── deploy.yml
├── app/
│   ├── emotion_model/
│   ├── lambda_function.py
│   ├── main.py
│   └── templates/
├── tests/
│   ├── test_api.py
│   ├── test_model.py
├── Dockerfile
├── buildspec.yml
├── fine_tune.py
├── requirements.txt
├── README.md

```

## Installation (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/LukeADay/emotion-detection.git
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

### 4. Run the Application Locally

You can run the Flask application locally for development:

```bash
cd app
python main.py
```

The API will be available at `http://127.0.0.1:8000`.

---

## AWS Deployment

### Prerequisites
- AWS CLI is configured with appropriate permissions.
- An S3 bucket is created to store model artifacts.

### Deployment Steps

1. Push changes to the repository.
2. The GitHub Actions workflow will:
    - Test the application.
    - Trigger AWS CodeBuild to:
        + Build a Docker container compatible with AWS Lambda.
        + Upload the container to an S3 bucket.
        + Deploy the Lambda function.
        + Update the API Gateway deployment.

###  API Gateway Endpoint

After deployment, the API will be available at: `https://9fuctupzoj.execute-api.eu-west-2.amazonaws.com/prod/predict`.

--- 

## Testing the Deployed API

### Make a POST request

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so happy today!"}' \
  https://9fuctupzoj.execute-api.eu-west-2.amazonaws.com/prod/predict
```

Expected Response:

```json
{"emotion": "joy"}
```

---

## CI/CD Workflow

The GitHub Actions workflow (`deploy.yml`) automates:

1. Running unit tests with `pytest`.
2. Triggering AWS CodeBuild to:
    - Build and package the application using the Dockerfile.
    - Deploy the Lambda function.
3. Updating the API Gateway deployment.

---

## Technologies Used

- Hugging Face Transformers: For the fine-tuned `DistilBERT` model.
- AWS Lambda: For serverless deployment.
- AWS CodeBuild: For containerizing and deploying the application.
- API Gateway: For routing requests.
- GitHub Actions: For CI/CD pipeline.
- Docker: For containerised deployment and local testing.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

