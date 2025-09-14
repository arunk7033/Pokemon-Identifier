### Overview

This project allows you to classify Pokémon from images using a deep learning model trained on an annotated Pokémon dataset. The model is capable of identifying the name of the Pokémon based on the input image.

### Prerequisites

1. Python 3.7 or higher
2. Install the required Python libraries by running:
    ```bash
    pip install -r requirements.txt
    ```
    Ensure that the `requirements.txt` file contains all necessary dependencies such as TensorFlow, NumPy, Pandas, and scikit-learn.

3. Place your Pokémon images in the `data/images` folder and ensure the labels are correctly annotated in `data/pokemon_labels.csv`.

### Training the Model

If you want to train the model from scratch, follow these steps:

1. Ensure the dataset is correctly placed in the `data` folder.
2. Run the following command to train the model:
    ```bash
    python3 model_training.py
    ```
3. The trained model will be saved as `pokemon_classifier.h5` in the project directory.

### Using the Pre-trained Model

To classify a Pokémon image using the pre-trained model:

1. Place the image you want to classify in the project directory or provide the correct path.
2. Run the following command:
    ```bash
    python3 pokemon_classification.py
    ```
3. The script will output the predicted Pokémon name.

### Example

To classify an image named `Pokemon 5.jpeg`, ensure the image is in the project directory and run:
```bash
python3 pokemon_classification.py
```
The output will display the predicted Pokémon name.

### Notes

- Ensure the image dimensions are 64x64 pixels or let the script resize them automatically.
- If you encounter any issues, verify that the dataset and model paths are correct.
- For custom datasets, update the `data/pokemon_labels.csv` file and retrain the model.


