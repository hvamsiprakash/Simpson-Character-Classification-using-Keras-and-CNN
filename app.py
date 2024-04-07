# # import streamlit as st
# # import requests
# # from PIL import Image
# # from io import BytesIO
# # import numpy as np
# # import tensorflow.keras as keras

# # # Function to load images from the provided URLs
# # def load_images():
# #     title1_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title1.png'    
# #     title2_url = 'https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/images/title2.png'
# #     title1_image = Image.open(BytesIO(requests.get(title1_url).content))
# #     title2_image = Image.open(BytesIO(requests.get(title2_url).content))
# #     return title1_image, title2_image

# # # Load and display images as titles
# # title1_image, title2_image = load_images()

# # # Set page background color to black
# # st.markdown("""
# #     <style>
# #         body {
# #             background-color: black;
# #             color: white;
# #         }
# #     </style>
# # """, unsafe_allow_html=True)

# # # Display title images in the main interface
# # st.image(title1_image, use_column_width=True)
# # st.image(title2_image, use_column_width=True)

# # # Function to preprocess image
# # def preprocess_image(image):
# #     img = image.resize((64, 64))
# #     img = np.array(img)
# #     img = img.astype("float32") / 255.0
# #     return img

# # # Load the model
# # @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# # def load_model():
# #     try:
# #         model_url='https://github.com/hvamsiprakash/Simpson-Character-Classification-using-Keras-and-CNN/raw/main/models/model.h5'
# #         response = requests.get(model_url)
# #         response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
# #         model_path = 'model (1).h5'
# #         with open(model_path, 'wb') as f:
# #             f.write(response.content)
# #         model = keras.models.load_model(model_path)
# #         return model
# #     except Exception as e:
# #         st.error(f"Error loading model: {str(e)}")
# #         return None

# # model = load_model()

# # # Function to predict
# # def predict(image, model):
# #     processed_image = preprocess_image(image)
# #     prediction = model.predict(np.expand_dims(processed_image, axis=0))
# #     predicted_class = np.argmax(prediction)
# #     return predicted_class

# # # Map class indices to character names
# # map_characters = {
# #     0: 'abraham_grampa_simpson',
# #     1: 'apu_nahasapeemapetilon',
# #     2: 'bart_simpson',
# #     3: 'charles_montgomery_burns',
# #     4: 'chief_wiggum',
# #     5: 'comic_book_guy',
# #     6: 'edna_krabappel',
# #     7: 'homer_simpson',
# #     8: 'kent_brockman',
# #     9: 'krusty_the_clown',
# #     10: 'lisa_simpson',
# #     11: 'marge_simpson',
# #     12: 'milhouse_van_houten',
# #     13: 'moe_szyslak',
# #     14: 'ned_flanders',
# #     15: 'nelson_muntz',
# #     16: 'principal_skinner',
# #     17: 'sideshow_bob'
# # }

# # # Streamlit UI
# # st.sidebar.title("Project Description")
# # st.sidebar.markdown("""
# # This is a Simpson Character Classifier. 
# # It takes an image of a Simpsons character as input and predicts which character it is. 
# # The model used for prediction is a convolutional neural network (CNN) trained on a dataset of Simpsons characters.
# # """)

# # st.sidebar.title("Classes Predicted")
# # # Display the 17 classes the model can predict
# # for character_name in map_characters.values():
# #     st.sidebar.write(character_name)

# # st.title("Simpson Character Classifier")

# # # Uploaded file for prediction
# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None and model is not None:
# #     image = Image.open(uploaded_file)
# #     # Resize the uploaded image
# #     image = image.resize((200, 200))
# #     st.image(image, caption='Uploaded Image', use_column_width=True)
# #     prediction = predict(image, model)
# #     predicted_character = map_characters.get(prediction, "Unknown")
    
# #     # Check for mismatches
# #     st.write(f"Predicted Character: {predicted_character}")




# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2

# # Load the pre-trained model
# model = load_model('models/model.h5')

# # Define character mapping
# map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
#                   3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 
#                   6: 'edna_krabappel', 7: 'homer_simpson', 8: 'kent_brockman', 
#                   9: 'krusty_the_clown', 10: 'lisa_simpson', 11: 'marge_simpson', 
#                   12: 'milhouse_van_houten', 13: 'moe_szyslak', 14: 'ned_flanders', 
#                   15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

# # Function to preprocess the uploaded image
# def preprocess_img(uploaded_file):
#     # Read the uploaded file as bytes, then decode into an image array
#     image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
#     # Resize the image to 64x64 pixels
#     resized_img = cv2.resize(image, (64, 64))
#     # Reshape the image to the format the model expects: (1, 64, 64, 3)
#     img_reshape = resized_img.reshape(1, 64, 64, 3)
#     return img_reshape

# # Function to predict the character from the uploaded image
# def predict_result(predict):
#     # Use the model to make a prediction
#     pred = model.predict(predict)
#     # Find the character with the highest prediction probability
#     character = map_characters[np.argmax(pred[0])]
#     # Format the character name for display
#     return character.replace('_', ' ').title()

# def main():
#     st.title("Simpsons Character Classifier")

#     # Upload image widget
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

#         # Predict button
#         if st.button('Predict'):
#             try:
#                 # Preprocess the uploaded image
#                 img = preprocess_img(uploaded_file)
#                 # Make a prediction
#                 pred = predict_result(img)
#                 # Display the prediction result
#                 st.success(f"The predicted character is: {pred}")
#             except Exception as e:
#                 st.error(f"Error processing file: {e}")

# if __name__ == "__main__":
#     main()



import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = load_model(model_path)
    return model

# Use lambda function to call load_model with a dummy argument
model_path = 'models/model.h5'
model = load_model(model_path)  # This will call load_model without passing any arguments

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    return img

# Function to make prediction
def predict_character(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    character_index = np.argmax(prediction)
    characters = {
        0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'
    }
    character_name = characters[character_index]
    return character_name.replace('_', ' ').title()

# Streamlit app
def main():
    st.title("Simpsons Character Predictor")
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        if st.button("Predict"):
            character = predict_character(image)
            st.success(f"The character is: {character}")

if __name__ == '__main__':
    main()
