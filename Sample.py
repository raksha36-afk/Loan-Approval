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
