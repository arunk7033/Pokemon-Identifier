import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Load the label encoder and the pre-trained model
print("Loading label encoder and model...")
labels_df = pd.read_csv('data/pokemon_labels.csv')
label_encoder = LabelEncoder()
label_encoder.fit(labels_df['label'])  # Fit the label encoder with the labels from the CSV
model = load_model('pokemon_classifier.h5')

print("Label encoder and model loaded successfully.")

# Step 2: Prediction function
def predict_pokemon(image_path):
    print(f"Predicting Pokémon for image: {image_path}")
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
    
    print(f"Prediction completed. Predicted Pokémon: {predicted_label[0]}")
    return predicted_label[0]

# Example: Predict a Pokémon image
image_path = 'Pokemon 5.jpeg'
predicted_pokemon = predict_pokemon(image_path)
print(f'Predicted Pokémon: {predicted_pokemon}')