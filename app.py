from flask import Flask, render_template, request, jsonify
import csv

app = Flask(__name__)

# Function to read substitutions from the CSV file
def get_ingredient_substitutes(ingredient):
    substitutes = {}
    with open('ingredient_substitutions_expanded.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            substitutes[row['Original Ingredient'].lower()] = row['Substitutes'].split(", ")
    # Return the substitutes for the given ingredient
    ingredient = ingredient.lower().strip()
    return substitutes.get(ingredient, ["No substitutes found for this ingredient."])

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/substitute', methods=['POST'])
def substitute():
    ingredient = request.form.get('ingredient')
    substitutes = get_ingredient_substitutes(ingredient)
    return render_template('results.html', ingredient=ingredient, substitutes=substitutes)

if __name__ == '__main__':
    app.run(debug=True)