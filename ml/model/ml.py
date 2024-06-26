# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
# Assuming the dataset is in a CSV file named 'loan_approval_data.csv'
data = pd.read_csv('D:/study materials/SEM-6/flaskblog/Loan-Approval/ml/model/loan_approval_dataset.csv')

#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data[" education"] = le.fit_transform(data[" education"])
data[" self_employed"] = le.fit_transform(data[" self_employed"])
data[" loan_status"] = le.fit_transform(data[" loan_status"])

# Define features and target variable
X = data.drop(columns=['loan_id', ' loan_status'])
y = data[' loan_status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')

# Save the Random Forest model to a pickle file
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Train an XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}')

# Save the XGBoost model to a pickle file
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# Import necessary libraries
import lime
import lime.lime_tabular
import numpy as np

# Initialize the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Not Approved', 'Approved'],
    mode='classification'
)

# Select a few samples from the test set for explanation
samples_to_explain = X_test.sample(5, random_state=42)

# Generate LIME explanations for these samples
lime_explanations = []
for idx in samples_to_explain.index:
    explanation = explainer.explain_instance(
        data_row=X_test.loc[idx],
        predict_fn=rf_model.predict_proba
    )
    lime_explanations.append(explanation)

# Display the explanations
for i, explanation in enumerate(lime_explanations):
    print(f'\nExplanation for sample {i + 1} (Index {samples_to_explain.index[i]}):')
    explanation.show_in_notebook(show_table=True)

# Import necessary libraries for FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained FinBERT model
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def get_finbert_explanation(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs

# Generate textual explanations for LIME values using FinBERT
for i, explanation in enumerate(lime_explanations):
    # Assuming explanation.as_list() returns a list of tuples (feature, weight)
    explanation_text = ' '.join([f'{feature} is important with a weight of {weight:.2f}.' for feature, weight in explanation.as_list()])
    sentiment = get_finbert_explanation(explanation_text)
    
    print(f'\nTextual Explanation for sample {i + 1} (Index {samples_to_explain.index[i]}):')
    print(explanation_text)
    print(f'Sentiment: {sentiment}')
