import numpy as np
from PIL import Image
import tensorflow as tf

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((64, 64))
    img = np.array(img)
    img = img.astype("float32") / 255.0
    return img

# Function to load the model
def load_model():
    model = tf.keras.models.load_model('models/model.h5')
    return model

# Function to predict character
def predict_character(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index
