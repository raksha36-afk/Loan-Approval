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

# Load the data
# Assuming your data is in a CSV file, replace 'your_dataset.csv' with your actual file name
data = pd.read_csv('../model/loan_approval_dataset.csv')

# Preprocessing
# Drop the 'loan_id' column as it is not needed for training
data.drop('loan_id', axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
for column in [' education', ' self_employed']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Encode the target variable
data[' loan_status'] = LabelEncoder().fit_transform(data[' loan_status'])

# Separate features and target variable
X = data.drop(' loan_status', axis=1)
y = data[' loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_test.shape)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Importing the module for LimeTabularExplainer
from lime import lime_tabular

# # Function to ensure the model output is in the expected format for LIME
def predict_proba(data):
    proba = model.predict(data)
    return np.hstack([1 - proba, proba])

 
# Instantiating the explainer object by passing in the training set,
# and the extracted features
explainer_lime = lime_tabular.LimeTabularExplainer(X_train,
                                                   feature_names=X.columns,
                                                   verbose=True, 
                                                   mode='classification')

# Load the model
model = load_model('../model/mlp_loan_approval_model.h5')

# Index corresponding to the test vector
i = 10
 
# Number denoting the top features
k = 11
 
# Calling the explain_instance method by passing in the:
#    1) ith test vector
#    2) prediction function used by our prediction model('reg' in this case)
#    3) the top features which we want to see, denoted by k
 
exp_lime = explainer_lime.explain_instance(
    X_test[i], predict_proba, num_features=k)

import matplotlib.pyplot as plt
# Visualize as an image
fig = exp_lime.as_pyplot_figure()
plt.show()

print(exp_lime.as_map())
# Visualize as a list
explanation_list = exp_lime.as_list()
print("Explanation as list of top features:")
for feature, weight in explanation_list:
    print(f"{feature}: {weight}")

no_of_dependents = 1
education = 'Graduate'
self_employed = 'Yes'
income_annum = 800000
loan_amount = 220
loan_term = 20
cibil_score = 750
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

# Encode categorical variables
label_encoders = {}
for column in [' education', ' self_employed']:
    label_encoders[column] = LabelEncoder()
    input_data[column] = label_encoders[column].fit_transform(input_data[column])

# Normalize the features
input_data = scaler.transform(input_data)

print(input_data)

exp_lime = explainer_lime.explain_instance(
    input_data[0], predict_proba, num_features=k)

import matplotlib.pyplot as plt
# Visualize as an image
fig = exp_lime.as_pyplot_figure()
plt.show()

decision =  'Approved' if(model.predict(input_data) > 0.5).astype("int32") else 'Rejected'
print(decision)
# Visualize as a list
explanation_list = exp_lime.as_list()
print("Explanation as list of top features:")
for feature, weight in explanation_list:
    print(f"{feature}: {weight}")
