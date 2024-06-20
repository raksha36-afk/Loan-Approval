import pytest
from flask import Flask
from flask.testing import FlaskClient
import pandas as pd
import sys
import os

# Add the path to the predictor module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flaskapp')))

# Import the Flask app and predictor classes
from main import app  # Ensure this path is correct based on your directory structure
# Add the path to the predictor module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml', 'prediction')))
from predictor import RandomForestPredictor, XGBoostPredictor

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client: FlaskClient):
    """Test the home page."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Loan Approval Prediction' in rv.data

def test_prediction(client: FlaskClient):
    """Test the prediction functionality."""
    data = {
        'no_of_dependents': 2,
        'education': 'Not Graduate',
        'self_employed': 'Yes',
        'income_annum': 4100000,
        'loan_amount': 200000,
        'loan_term': 8,
        'cibil_score': 850,
        'residential_assets_value': 2700000,
        'commercial_assets_value': 2200000,
        'luxury_assets_value': 8800000,
        'bank_asset_value': 3300000
    }
    
    rv = client.post('/predict', data=data)
    assert rv.status_code == 200
    assert b'Random Forest Prediction: Approved' in rv.data or b'Random Forest Prediction: Rejected' in rv.data
    assert b'XGBoost Prediction: Approved' in rv.data or b'XGBoost Prediction: Rejected' in rv.data
