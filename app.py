import os
# import keras
import tensorflow as tf

import streamlit as st
import numpy as np
import glob
from PIL import Image

st.header("Flower Recommendation")

flower_names = ['astilbe',
 'bellflower',
 'black_eyed_susan',
 'calendula',
 'california_poppy',
 'carnation',
 'common_daisy',
 'coreopsis',
 'dandelion',
 'iris',
 'rose',
 'sunflower',
 'tulip',
 'water_lily']

model = tf.keras.models.load_model("Flower_Recognition_Model.h5")

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(256,256))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    return flower_names[np.argmax(result)]

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file,width=200)

    predict_flower = classify_images(uploaded_file)
    images_folder = os.path.join('train', predict_flower)
    images_to_show = glob.glob(os.path.join(images_folder, '*.jpg'))[:8]
    if images_to_show:
        st.write(f"Displaying images from the category: {predict_flower}")
        for image_path in images_to_show:
            image = Image.open(image_path)
            st.image(image, caption=os.path.basename(image_path), use_container_width=True, width=100)
    else:
        st.write("No images found for the classified category.")
