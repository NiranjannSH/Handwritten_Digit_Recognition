import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model(
    "handwritten_digit_recognition_model.h5", compile=False)

# Function to preprocess the uploaded image


def preprocess_image(image):
    # Resize the image to 28x28 and convert to grayscale
    img = image.resize((28, 28)).convert('L')
    # Convert image to numpy array
    img_array = np.array(img)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Reshape the image to match model input shape
    img_array = img_array.reshape((-1, 28, 28, 1))
    return img_array

# Streamlit app


def main():
    st.title("Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit (0-9) and click 'Predict'.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=250)
        # Preprocess the image
        processed_image = preprocess_image(image)

        if st.button('Predict'):
            # Make prediction
            prediction = model.predict(processed_image)
            # Get the predicted digit
            predicted_digit = np.argmax(prediction)
            st.write(f"Prediction: {predicted_digit}")


if __name__ == '__main__':
    main()
