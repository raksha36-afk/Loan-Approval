# Import necessary libraries
import pandas as pd
import pickle
from abc import ABC, abstractmethod

# Abstract base class
class LoanApprovalPredictor(ABC):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    @abstractmethod
    def preprocess(self, input_data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

# Concrete class for Random Forest
class RandomForestPredictor(LoanApprovalPredictor):
    def preprocess(self, input_data):
        input_data[' education'] = input_data[' education'].astype('category').cat.codes
        input_data[' self_employed'] = input_data[' self_employed'].astype('category').cat.codes
        return input_data

    def predict(self, input_data):
        preprocessed_data = self.preprocess(input_data)
        prediction = self.model.predict(preprocessed_data)
        return prediction

# Concrete class for XGBoost
class XGBoostPredictor(LoanApprovalPredictor):
    def preprocess(self, input_data):
        input_data[' education'] = input_data[' education'].astype('category').cat.codes
        input_data[' self_employed'] = input_data[' self_employed'].astype('category').cat.codes
        return input_data

    def predict(self, input_data):
        preprocessed_data = self.preprocess(input_data)
        prediction = self.model.predict(preprocessed_data)
        return prediction

# Import necessary libraries
import pandas as pd
from predictor import RandomForestPredictor, XGBoostPredictor

import pandas as pd
from predictor import RandomForestPredictor, XGBoostPredictor

# Example input data
input_data = pd.DataFrame({
    ' no_of_dependents': [2],
    ' education': ['Not Graduate'],
    ' self_employed': ['Yes'],
    ' income_annum': [4100000],
    ' loan_amount': [200000],
    ' loan_term': [8],
    ' cibil_score': [850],
    ' residential_assets_value': [2700000],
    ' commercial_assets_value': [2200000],
    ' luxury_assets_value': [8800000],
    ' bank_asset_value': [3300000]
})

if __name__ == "__main__":
    # Predict using Random Forest
    rf_predictor = RandomForestPredictor('../model/rf_model.pkl')
    rf_prediction = rf_predictor.predict(input_data)
    rf_status = ['Approved' if pred == 0 else 'Rejected' for pred in rf_prediction]
    print(f'Random Forest Prediction: {rf_status}')

    # Predict using XGBoost
    xgb_predictor = XGBoostPredictor('../model/xgb_model.pkl')
    xgb_prediction = xgb_predictor.predict(input_data)
    xgb_status = ['Approved' if pred == 0 else 'Rejected' for pred in xgb_prediction]
    print(f'XGBoost Prediction: {xgb_status}')


