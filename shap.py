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

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load the model
model = load_model('../model/mlp_loan_approval_model.h5')

import shap

# Initialize the SHAP Deep Explainer
explainer_shap = shap.Explainer(model, X_train[:100])

# Index corresponding to the test vector for local explanation
i = 10

# Compute SHAP values for the ith instance in X_test
shap_values = explainer_shap.shap_values(X_test[i:i+1])
print(shap_values)

# # Visualize local explanation
# shap.initjs()
# shap.force_plot(explainer_shap.expected_value, shap_values, X_test[i])

# # Summarize global explanations using SHAP summary plot
# shap_values_summary = explainer_shap.shap_values(X_test)
# print(shap_values_summary)
# shap.summary_plot(shap_values_summary, X_test)

shap_values2 = explainer_shap(X_test[i:i+1])
# Waterfall plot
shap.plots.waterfall(shap_values2[0])

# Beeswarm plot for a single instance
shap.plots.beeswarm(shap_values2, max_display=10)

# Plot bar chart for single instance
shap.plots.bar(shap_values2[0])

# Force plot
# Force plot and save as image
plt.figure()
shap.plots.force(shap_values2[0], matplotlib=True,feature_names=X.columns.tolist(),show=False)
plt.savefig('force_plot.png')
plt.close()
#bbox_inches='tight'
