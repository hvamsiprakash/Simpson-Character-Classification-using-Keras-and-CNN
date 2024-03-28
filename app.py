# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import tensorflow.keras as keras

# # Function to load images from the provided URLs
# def load_images():
#     title1_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title1.png'    
#     title2_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title2.png'
#     title1_image = Image.open(BytesIO(requests.get(title1_url).content))
#     title2_image = Image.open(BytesIO(requests.get(title2_url).content))
#     return title1_image, title2_image

# # Load and display images as titles
# title1_image, title2_image = load_images()

# # Set page background color to black
# st.markdown("""
#     <style>
#         body {
#             background-color: black;
#             color: white;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Display title images in the main interface
# st.image(title1_image, use_column_width=True)
# st.image(title2_image, use_column_width=True)

# # Function to preprocess image
# def preprocess_image(image):
#     img = image.resize((64, 64))
#     img = np.array(img)
#     img = img.astype("float32") / 255.0
#     return img

# # Load the model
# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def load_model():
#     try:
#         model_url='https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/models/model.h5'
#         response = requests.get(model_url)
#         response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
#         model_path = 'model (1).h5'
#         with open(model_path, 'wb') as f:
#             f.write(response.content)
#         model = keras.models.load_model(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# model = load_model()

# # Function to predict
# def predict(image, model):
#     processed_image = preprocess_image(image)
#     prediction = model.predict(np.expand_dims(processed_image, axis=0))
#     predicted_class = np.argmax(prediction)
#     return predicted_class

# # Map class indices to character names
# map_characters = {
#     0: 'abraham_grampa_simpson',
#     1: 'apu_nahasapeemapetilon',
#     2: 'bart_simpson',
#     3: 'charles_montgomery_burns',
#     4: 'chief_wiggum',
#     5: 'comic_book_guy',
#     6: 'edna_krabappel',
#     7: 'homer_simpson',
#     8: 'kent_brockman',
#     9: 'krusty_the_clown',
#     10: 'lisa_simpson',
#     11: 'marge_simpson',
#     12: 'milhouse_van_houten',
#     13: 'moe_szyslak',
#     14: 'ned_flanders',
#     15: 'nelson_muntz',
#     16: 'principal_skinner',
#     17: 'sideshow_bob'
# }

# # Streamlit UI
# st.sidebar.title("Project Description")
# st.sidebar.markdown("""
# This is a Simpson Character Classifier. 
# It takes an image of a Simpsons character as input and predicts which character it is. 
# The model used for prediction is a convolutional neural network (CNN) trained on a dataset of Simpsons characters.
# """)

# st.sidebar.title("Classes Predicted")
# # Display the 17 classes the model can predict
# for character_name in map_characters.values():
#     st.sidebar.write(character_name)

# st.title("Simpson Character Classifier")

# # Uploaded file for prediction
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None and model is not None:
#     image = Image.open(uploaded_file)
#     # Resize the uploaded image
#     image = image.resize((200, 200))
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     prediction = predict(image, model)
#     predicted_character = map_characters.get(prediction, "Unknown")
    
#     # Check for mismatches
#     st.write(f"Predicted Character: {predicted_character}")



import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

# Function to load images from the provided URLs
def load_images():
    title1_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title1.png'    
    title2_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title2.png'
    title1_image = Image.open(BytesIO(requests.get(title1_url).content))
    title2_image = Image.open(BytesIO(requests.get(title2_url).content))
    return title1_image, title2_image

# Load and display images as titles
title1_image, title2_image = load_images()

# Set page background color to black
st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Display title images in the main interface
st.image(title1_image, use_column_width=True)
st.image(title2_image, use_column_width=True)

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((64, 64))
    img = np.array(img)
    img = img.astype("float32") / 255.0
    return img

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('models/model.h5')
    return model 

model = load_model()

# Map class indices to character names
map_characters = {
    0: 'abraham_grampa_simpson',
    1: 'apu_nahasapeemapetilon',
    2: 'bart_simpson',
    3: 'charles_montgomery_burns',
    4: 'chief_wiggum',
    5: 'comic_book_guy',
    6: 'edna_krabappel',
    7: 'homer_simpson',
    8: 'kent_brockman',
    9: 'krusty_the_clown',
    10: 'lisa_simpson',
    11: 'marge_simpson',
    12: 'milhouse_van_houten',
    13: 'moe_szyslak',
    14: 'ned_flanders',
    15: 'nelson_muntz',
    16: 'principal_skinner',
    17: 'sideshow_bob'
}

# Streamlit UI
st.sidebar.title("Project Description")
st.sidebar.markdown("""
This is a Simpson Character Classifier. 
It takes an image of a Simpsons character as input and predicts which character it is. 
The model used for prediction is a convolutional neural network (CNN) trained on a dataset of Simpsons characters.
""")

st.sidebar.title("Classes Predicted")
# Display the 17 classes the model can predict
for character_name in map_characters.values():
    st.sidebar.write(character_name)

st.title("Simpson Character Classifier")

# Uploaded file for prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    # Resize the uploaded image
    image = image.resize((64, 64))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image and predict character
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_character = map_characters.get(predicted_class_index, "Unknown")
    
    # Display predicted character
    st.write(f"Predicted Character: {predicted_character}")

# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import tensorflow as tf
# from prediction import load_model, predict_character

# # Function to load images from the provided URLs
# def load_images():
#     title1_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title1.png'    
#     title2_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title2.png'
#     title1_image = Image.open(BytesIO(requests.get(title1_url).content))
#     title2_image = Image.open(BytesIO(requests.get(title2_url).content))
#     return title1_image, title2_image

# # Load and display images as titles
# title1_image, title2_image = load_images()

# # Set page background color to black
# st.markdown("""
#     <style>
#         body {
#             background-color: black;
#             color: white;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Display title images in the main interface
# st.image(title1_image, use_column_width=True)
# st.image(title2_image, use_column_width=True)

# # Streamlit UI
# st.sidebar.title("Project Description")
# st.sidebar.markdown("""
# This is a Simpson Character Classifier. 
# It takes an image of a Simpsons character as input and predicts which character it is. 
# The model used for prediction is a convolutional neural network (CNN) trained on a dataset of Simpsons characters.
# """)

# # Function to load model
# @st.cache(allow_output_mutation=True)
# def load_cached_model():
#     return load_model()

# model = load_cached_model()

# # Map class indices to character names
# map_characters = {
#     0: 'abraham_grampa_simpson',
#     1: 'apu_nahasapeemapetilon',
#     2: 'bart_simpson',
#     3: 'charles_montgomery_burns',
#     4: 'chief_wiggum',
#     5: 'comic_book_guy',
#     6: 'edna_krabappel',
#     7: 'homer_simpson',
#     8: 'kent_brockman',
#     9: 'krusty_the_clown',
#     10: 'lisa_simpson',
#     11: 'marge_simpson',
#     12: 'milhouse_van_houten',
#     13: 'moe_szyslak',
#     14: 'ned_flanders',
#     15: 'nelson_muntz',
#     16: 'principal_skinner',
#     17: 'sideshow_bob'
# }

# st.title("Simpson Character Classifier")

# # Uploaded file for prediction
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None and model is not None:
#     image = Image.open(uploaded_file)
#     # Resize the uploaded image
#     image = image.resize((64, 64))
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Predict character
#     predicted_class_index = predict_character(image, model)
#     predicted_character = map_characters.get(predicted_class_index, "Unknown")
    
#     # Display predicted character
#     st.write(f"Predicted Character: {predicted_character}")

