import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from lime import lime_tabular

# Load the data
data = pd.read_csv('../model/loan_approval_dataset.csv')

# Preprocessing
data.drop('loan_id', axis=1, inplace=True)

label_encoders = {}
for column in [' education', ' self_employed']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

data[' loan_status'] = LabelEncoder().fit_transform(data[' loan_status'])

X = data.drop(' loan_status', axis=1)
y = data[' loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#abc
def predict_proba(data):
    proba = model.predict(data)
    return np.hstack([1 - proba, proba])

explainer_lime = lime_tabular.LimeTabularExplainer(X_train,
                                                   feature_names=X.columns,
                                                   verbose=True, 
                                                   mode='classification')

model = load_model('../model/mlp_loan_approval_model.h5')

i = 10
k = 11

exp_lime = explainer_lime.explain_instance(X_test[i], predict_proba, num_features=k)

# Function to clean feature names in LIME explanation
def clean_feature_names(exp):
    exp.domain_mapper.feature_names = [feature.split(">")[0].split("< ")[0].strip() for feature in exp.domain_mapper.feature_names]

clean_feature_names(exp_lime)

fig = exp_lime.as_pyplot_figure()
plt.show()

exp_map = exp_lime.as_map()
class_1_explanations = exp_map[1]  # Assuming binary classification, focusing on class 1

feature_names = X.columns
features_weights = [(feature_names[feature_id], weight) for feature_id, weight in class_1_explanations]

print("Explanation as list of top features:")
for feature, weight in features_weights:
    print(f"{feature}: {weight}")

# Custom input data
no_of_dependents = 123
education = ' Graduate'
self_employed = ' Yes'
income_annum = 800000
loan_amount = 220000000000000
loan_term = 20
cibil_score = 75
residential_assets_value = 1300000
commercial_assets_value = 800000
luxury_assets_value = 2800000
bank_asset_value = 600000

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

for column in [' education', ' self_employed']:
    input_data[column] = label_encoders[column].transform(input_data[column])

input_data = scaler.transform(input_data)

exp_lime = explainer_lime.explain_instance(input_data[0], predict_proba, num_features=k)

clean_feature_names(exp_lime)

fig = exp_lime.as_pyplot_figure()
plt.show()

decision = 'Approved' if (model.predict(input_data) > 0.5).astype("int32") else 'Rejected'
print(decision)

exp_map = exp_lime.as_map()
class_1_explanations = exp_map[1]

features_weights = [(feature_names[feature_id], weight) for feature_id, weight in class_1_explanations]

print("Explanation as list of top features:")
for feature, weight in features_weights:
    print(f"{feature}: {weight}")

# Extract feature names and weights for plotting
features = [feature.split(' ')[0] for feature, weight in explanation_list]
weights = [weight for feature, weight in explanation_list]

# Plotting the bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(features, weights, color='skyblue')
plt.xlabel('Impact on Prediction')
plt.title('Impact of Top Features on Model Prediction for Sample 10')

# Adding data labels
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', ha='left', va='center')

plt.show()

import spacy

# Example LIME weights (replace with your actual LIME weights)
lime_weights = {
    'income': 0.6,
    'age': -0.3,
    'credit_score': 0.4,
    'employment_status': 0.2,
    'loan_amount': 0.5
}

# Explanation templates
approval_template = "Congratulations! Your loan has been approved because of your {features}."
rejection_template = "We regret to inform you that your loan application was rejected due to your {features}."

# Function to generate explanation based on weights
def generate_explanation(weights):
    reasons = []
    for feature, weight in weights.items():
        if weight > 0:
            reasons.append(f"high {feature}")
        else:
            reasons.append(f"low {feature}")
    
    features_text = ", ".join(reasons)
    
    if sum(weights.values()) > 0:
        explanation = approval_template.format(features=features_text)
    else:
        explanation = rejection_template.format(features=features_text)
    
    return explanation

# Generate and print explanation
explanation = generate_explanation(lime_weights)
print(explanation)

# Optionally, use SpaCy for more advanced text processing tasks
nlp = spacy.load('en_core_web_sm')
doc = nlp(explanation)

# Access SpaCy's linguistic annotations
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
