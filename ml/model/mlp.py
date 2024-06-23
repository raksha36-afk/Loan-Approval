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
data = pd.read_csv('loan_approval_dataset.csv')

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

# Define the MLP model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predicting the results
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Classification Report
print(classification_report(y_test, y_pred))

# Plotting Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save('mlp_loan_approval_model.h5')
print("Model saved as mlp_loan_approval_model.h5")

# # Importing the module for LimeTabularExplainer
# from lime import lime_tabular
 
# # Instantiating the explainer object by passing in the training set,
# # and the extracted features
# explainer_lime = lime_tabular.LimeTabularExplainer(X_train,
#                                                    feature_names=X.columns,
#                                                    verbose=True, 
#                                                    mode='regression')

# # Load the model
# model = load_model('mlp_loan_approval_model.h5')

# # Index corresponding to the test vector
# i = 10
 
# # Number denoting the top features
# k = 5
 
# # Calling the explain_instance method by passing in the:
# #    1) ith test vector
# #    2) prediction function used by our prediction model('reg' in this case)
# #    3) the top features which we want to see, denoted by k
 
# exp_lime = explainer_lime.explain_instance(
#     X_test[i], model.predict, num_features=k)

# import matplotlib.pyplot as plt
# # Visualize as an image
# fig = exp_lime.as_pyplot_figure()
# plt.show()

# # Visualize as a list
# explanation_list = exp_lime.as_list()
# print("Explanation as list of top features:")
# for feature, weight in explanation_list:
#     print(f"{feature}: {weight}")

# no_of_dependents = 0
# education = 'Graduate'
# self_employed = 'Yes'
# income_annum = 800000
# loan_amount = 2200000
# loan_term = 20
# cibil_score = 782
# residential_assets_value = 1300000
# commercial_assets_value = 800000
# luxury_assets_value = 2800000
# bank_asset_value = 600000

# input_data = pd.DataFrame({
#         ' no_of_dependents': [no_of_dependents],
#         ' education': [education],
#         ' self_employed': [self_employed],
#         ' income_annum': [income_annum],
#         ' loan_amount': [loan_amount],
#         ' loan_term': [loan_term],
#         ' cibil_score': [cibil_score],
#         ' residential_assets_value': [residential_assets_value],
#         ' commercial_assets_value': [commercial_assets_value],
#         ' luxury_assets_value': [luxury_assets_value],
#         ' bank_asset_value': [bank_asset_value]
#     })

# # Encode categorical variables
# label_encoders = {}
# for column in [' education', ' self_employed']:
#     label_encoders[column] = LabelEncoder()
#     input_data[column] = label_encoders[column].fit_transform(input_data[column])

# # Normalize the features
# input_data = scaler.transform(input_data)


# exp_lime = explainer_lime.explain_instance(
#     input_data[0], model.predict, num_features=k)

# import matplotlib.pyplot as plt
# # Visualize as an image
# fig = exp_lime.as_pyplot_figure()
# plt.show()

# print('Rejected' if(model.predict(input_data) > 0.5).astype("int32") else 'Approved')
# # Visualize as a list
# explanation_list = exp_lime.as_list()
# print("Explanation as list of top features:")
# for feature, weight in explanation_list:
#     print(f"{feature}: {weight}")

# from transformers import BertTokenizer, BertLMHeadModel, pipeline

# # Initialize tokenizer and model for BertLMHeadModel
# finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# finbert_model = BertLMHeadModel.from_pretrained('yiyanghkust/finbert-tone')

# # Create a text generation pipeline
# finbert_nlp = pipeline('text-generation', model=finbert_model, tokenizer=finbert_tokenizer)

# # Function to generate human-readable rejection explanation
# def generate_rejection_explanation(weights):
#     input_text = "Your loan application was evaluated based on several factors. "
#     rejection_reasons = []
    
#     # Identify and sort features by their impact (negative weights)
#     sorted_features = sorted(weights, key=lambda x: x[1])
    
#     for feature, weight in sorted_features:
#         if weight < 0:
#             rejection_reasons.append(feature)
    
#     if rejection_reasons:
#         input_text += "Due to "
#         for idx, reason in enumerate(rejection_reasons):
#             if idx > 0:
#                 input_text += ", "
#             input_text += f"{reason}"
#         input_text += ", your loan application has been rejected."
#     else:
#         input_text = "Your loan application has been approved."

#     explanation = finbert_nlp(input_text, max_length=100, num_return_sequences=1)
#     return explanation[0]['generated_text']

# # Example usage
# rejection_explanation = generate_rejection_explanation(explanation_list)
# print("Rejection Explanation:", rejection_explanation)
