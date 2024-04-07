import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the pre-trained model
model = load_model('models/model.h5')

# Define character mapping
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
                  3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 
                  6: 'edna_krabappel', 7: 'homer_simpson', 8: 'kent_brockman', 
                  9: 'krusty_the_clown', 10: 'lisa_simpson', 11: 'marge_simpson', 
                  12: 'milhouse_van_houten', 13: 'moe_szyslak', 14: 'ned_flanders', 
                  15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

# Function to preprocess the uploaded image
def preprocess_img(uploaded_file):
    # Read the uploaded file as bytes, then decode into an image array
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Resize the image to 64x64 pixels
    resized_img = cv2.resize(image, (64, 64))
    # Reshape the image to the format the model expects: (1, 64, 64, 3)
    img_reshape = resized_img.reshape(1, 64, 64, 3)
    # Normalize the pixel values to be between 0 and 1
    img_normalized = img_reshape.astype('float32') / 255.
    return img_normalized

# Function to predict the character from the uploaded image
def predict_result(predict):
    # Use the model to make a prediction
    pred = model.predict(predict)
    # Find the character with the highest prediction probability
    character = map_characters[np.argmax(pred[0])]
    # Format the character name for display
    return character.replace('_', ' ').title()

def main():
    st.title("Simpsons Character Classifier")

    # Upload image widget
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Predict button
        if st.button('Predict'):
            try:
                # Preprocess the uploaded image
                img = preprocess_img(uploaded_file)
                # Make a prediction
                pred = predict_result(img)
                # Display the prediction result
                st.success(f"The predicted character is: {pred}")
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
