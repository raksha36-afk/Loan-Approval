import spacy

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Function to generate explanation with enhanced text processing
def generate_explanation_enhanced(weights):
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
    
    # Perform enhanced text processing (e.g., using spaCy)
    doc = nlp(explanation)
    processed_text = " ".join([token.lemma_ for token in doc])
    
    return processed_text

# Generate and print enhanced explanation
enhanced_explanation = generate_explanation_enhanced(lime_weights)
print(enhanced_explanation)
