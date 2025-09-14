import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# Step 1: Load the CSV and the directory structure
print("Loading CSV data...")
labels_df = pd.read_csv('data/pokemon_labels.csv')
print(f"Loaded {len(labels_df)} records from CSV.")

# Step 2: Process the images and their labels
def load_data(labels_df):
    print("Processing images and labels...")
    images = []
    labels = []
    for index, row in labels_df.iterrows():
        img_path = row['id']
        label = row['label']
        
        # Load image
        print(f"Loading image: {img_path}")
        img = load_img(img_path, target_size=(64, 64))  # Resize to a fixed size
        img_array = img_to_array(img) / 255.0  # Normalize the image

        images.append(img_array)
        labels.append(label)

    print(f"Processed {len(images)} images.")
    return np.array(images), np.array(labels)

images, labels = load_data(labels_df)

# Step 3: Encode labels
print("Encoding labels...")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)  # One-hot encoding
print("Labels encoded successfully.")

# Step 4: Split the data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

# Step 5: Define the CNN model
print("Defining the CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Number of Pokémon classes
])
print("Model defined successfully.")

# Step 6: Compile the model
print("Compiling the model...")
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# Step 7: Train the model
print("Starting training...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32)
print("Training completed.")

# Step 8: Save the model
print("Saving the model...")
model.save('pokemon_classifier.h5')
print("Model saved successfully.")

# Step 9: Prediction function
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
image_path = 'data/images/Zapdos/26b19f8809ce496eae2e1b822d54492c.jpg'
predicted_pokemon = predict_pokemon(image_path)
print(f'Predicted Pokémon: {predicted_pokemon}')