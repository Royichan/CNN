import streamlit as st
from PIL import Image
import numpy as np
import requests

# Function to send features to an API for localization prediction
def apiRequest(imageArray):
    body = {"image":imageArray.tolist()}
    try:
        response = requests.post(url="http://localhost:105/riceDiseasePrediction", json=body)
    except requests.exceptions.ConnectionError as e:
        st.text("API Connection Failed")
        return None, None

    if response.status_code == 200:
        disease, confidence = response.json()['disease'], response.json()['confidence']
        return disease, confidence
    else:
        st.text("API Call Failed")

def main():
    st.title("Rice Disease Classification")
    uploadedFile = st.file_uploader("Choose the image", type=['png', 'jpg', 'jpeg'])    

    if st.button("Predict Disease"):
        if uploadedFile is not None:
            image = Image.open(uploadedFile)

            st.image(image)

            image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            print(image)
            disease, confidence = apiRequest(img_array)
            if disease != None:
                st.write(f'There is {confidence}% chance it is {disease}')
        else:
            st.write(f'Image was not uploaded')


if __name__ == "__main__":
    main()
