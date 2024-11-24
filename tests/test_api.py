import pytest
from app.main import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test the home route."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json == {"message": "Emotion Detection API is live!"}

def test_predict_route(client):
    """Test the predict route with valid input."""
    response = client.post(
        "/predict",
        json={"text": "I am so happy today!"},
    )
    assert response.status_code == 200
    assert "emotion" in response.json

def test_predict_no_text(client):
    """Test the predict route with missing text."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.json == {"error": "No text provided"}
