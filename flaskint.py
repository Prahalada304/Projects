from flask import Flask, render_template, request, jsonify
from tensorflow.python.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
#from tensorflow import *

app = Flask(__name__)

# Load the trained models
nn_model = load_model('ingredient_substitution_model.h5')
w2v_model = Word2Vec.load('ingredient_w2v.model')
ingredient_to_id = {ingredient: idx for idx, ingredient in enumerate(w2v_model.wv.index_to_key)}
id_to_ingredient = {idx: ingredient for ingredient, idx in ingredient_to_id.items()}

def predict_substitute(ingredient):
    if ingredient not in w2v_model.wv:
        return ["No substitutes found."]
    vector = w2v_model.wv[ingredient]
    prediction = nn_model.predict(np.array([vector]))
    top_indices = prediction[0].argsort()[-5:][::-1]
    substitutes = [id_to_ingredient[idx] for idx in top_indices]
    return substitutes

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/substitute', methods=['POST'])
def substitute():
    ingredient = request.form.get('ingredient').strip().lower()
    substitutes = predict_substitute(ingredient)
    return render_template('results.html', ingredient=ingredient, substitutes=substitutes)

if __name__ == '__main__':
    app.run(debug=True)
