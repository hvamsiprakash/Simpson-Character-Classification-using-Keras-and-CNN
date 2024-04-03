import streamlit as st
import numpy as np
import tensorflow as tf

@st.cache
def load_model_and_labels():
    model = tf.keras.models.load_model('models/model.h5')
    
    # Recompile the model to ensure its readiness
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    class_names = {}
    with open('models/labels.txt', 'r') as f:
        for line in f:
            try:
                key, val = line.strip().split(' ', 1)
                class_names[int(key)] = val.replace("_", " ")  # Replacing underscores with spaces
            except ValueError:
                # Skip lines that don't contain the expected format
                pass
    return model, class_names

def preprocess_image(image):
    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image
    img = image.resize((64, 64))
    
    # Convert image to numpy array
    img = np.array(img)
    
    # Normalize image
    img = img.astype("float32") / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
