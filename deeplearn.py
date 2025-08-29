import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM
from tensorflow.python.keras.utils import to_categorical
#from tensorflow import *

# Load the ingredient-substitution dataset
data = pd.read_csv('ingredient_substitutions_expanded.csv')  # A file with 'Ingredient' and 'Substitutes' columns

# Preprocessing
ingredients = list(set(data['Ingredient']).union(set(data['Substitutes'])))
ingredient_to_id = {ingredient: i for i, ingredient in enumerate(ingredients)}
id_to_ingredient = {i: ingredient for ingredient, i in ingredient_to_id.items()}

data['Ingredient_ID'] = data['Ingredient'].map(ingredient_to_id)
data['Substitute_ID'] = data['Substitutes'].map(ingredient_to_id)

# Train a Word2Vec model on the ingredient data
pairs = [[row['Ingredient'], row['Substitutes']] for _, row in data.iterrows()]
w2v_model = Word2Vec(pairs, vector_size=50, min_count=1, workers=4)

# Prepare the input and output for deep learning
X = np.array([w2v_model.wv[ingredient] for ingredient in data['Ingredient']])
y = to_categorical(data['Substitute_ID'], num_classes=len(ingredients))

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a neural network for predicting substitutes
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(len(ingredients), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save the model
model.save('ingredient_substitution_model.h5')
