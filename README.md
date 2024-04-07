# Simpson Character Classification 

This project implements a Simpson character classification system using Keras, TensorFlow, and Convolutional Neural Networks (CNNs). It predicts 17 classes of Simpson characters achieving an accuracy of 96.83% on the Simpson Characters dataset.

## Dataset 📸

The dataset used for training the model is available on [Kaggle](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset). It consists of images of various characters from the Simpsons animated TV series.

## Streamlit Demo 🚀

Try out the Simpson Character Classifier on Streamlit: [Simpson Character Classifier](https://simpson-character-classification-using-keras-and-cnn-4s9yvvz6v.streamlit.app/)

## Working 🛠️

1. **Data Preparation**: The dataset is loaded and preprocessed using OpenCV to resize images to a standard size of 64x64 pixels.

2. **Model Training**: Two CNN models are trained using Keras and TensorFlow. The first model has 4 convolutional layers, and the second model has 6 convolutional layers. Both models are trained with data augmentation techniques to improve performance.

3. **Evaluation**: The trained models are evaluated on a test set, and classification reports are generated to assess model performance.

4. **Prediction**: Users can upload an image of a Simpson character to the Streamlit app, and the model predicts the character's name based on the uploaded image.
