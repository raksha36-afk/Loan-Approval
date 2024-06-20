from flask import Flask, request, render_template
import pandas as pd
import sys
import os

# Add the path to the predictor module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml', 'prediction')))

from predictor import RandomForestPredictor, XGBoostPredictor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    no_of_dependents = int(request.form['no_of_dependents'])
    education = request.form['education']
    self_employed = request.form['self_employed']
    income_annum = int(request.form['income_annum'])
    loan_amount = int(request.form['loan_amount'])
    loan_term = int(request.form['loan_term'])
    cibil_score = int(request.form['cibil_score'])
    residential_assets_value = int(request.form['residential_assets_value'])
    commercial_assets_value = int(request.form['commercial_assets_value'])
    luxury_assets_value = int(request.form['luxury_assets_value'])
    bank_asset_value = int(request.form['bank_asset_value'])

    input_data = pd.DataFrame({
        ' no_of_dependents': [no_of_dependents],
        ' education': [education],
        ' self_employed': [self_employed],
        ' income_annum': [income_annum],
        ' loan_amount': [loan_amount],
        ' loan_term': [loan_term],
        ' cibil_score': [cibil_score],
        ' residential_assets_value': [residential_assets_value],
        ' commercial_assets_value': [commercial_assets_value],
        ' luxury_assets_value': [luxury_assets_value],
        ' bank_asset_value': [bank_asset_value]
    })

    rf_predictor = RandomForestPredictor('../ml/model/rf_model.pkl')
    rf_prediction = rf_predictor.predict(input_data)
    rf_status = ['Approved' if pred == 0 else 'Rejected' for pred in rf_prediction][0]

    xgb_predictor = XGBoostPredictor('../ml/model/xgb_model.pkl')
    xgb_prediction = xgb_predictor.predict(input_data)
    xgb_status = ['Approved' if pred == 0 else 'Rejected' for pred in xgb_prediction][0]

    return render_template('result.html', rf_status=rf_status, xgb_status=xgb_status)

if __name__ == '__main__':
    app.run(debug=True)
