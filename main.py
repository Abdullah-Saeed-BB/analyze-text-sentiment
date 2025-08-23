from flask import Flask, render_template, request, jsonify
import re
import numpy as np
import joblib

model = joblib.load("./estimators/model.joblib")
vectorizer = joblib.load("./estimators/vectorizer.joblib")

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
intercept = model.intercept_[0]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')

    X_text = vectorizer.transform([text])

    negative_proba = model.predict_proba(X_text)[0][0]

    tfidf_values = X_text.toarray()[0]

    contributions = (-intercept * (negative_proba * 2 - 1)) + (tfidf_values * coefs)

    # Only keep features present in the text
    nonzero_indices = np.where(tfidf_values != 0)[0]

    result = []
    # Map feature -> contribution
    feature_to_weight = {feature_names[i]: contributions[i] for i in nonzero_indices}

    # Split text into tokens for ordering
    tokens = re.findall(r'\b\w+\b', text)  
    
    # Collect unigrams + bigrams in order
    words = []
    i = 0
    while i < len(tokens):
        unigram = tokens[i]
        if i < len(tokens) - 1:
            bigram = tokens[i] + " " + tokens[i+1]
            if bigram.lower() in feature_to_weight:
                words.append(bigram)
                i += 2
                continue
        if unigram.lower() in feature_to_weight:
            words.append(unigram)
        else:
            words.append(unigram)
        i +=1 
    words_weight = [-feature_to_weight.get(token.lower(), 0) for token in words]
    result = list(zip(words, words_weight))

    return jsonify({'words': result, "negative_percent": negative_proba})

if __name__ == '__main__':
    app.run(debug=True)
