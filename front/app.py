import os
import torch
import streamlit as st
from PIL import Image
from utils import load_image_list
import requests
import io

def main():
    # Get image lists
    image_paths = load_image_list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Image & model selection
    image_slider, model_selector = st.columns([2, 1])
    with image_slider:
        # Select image
        image_slider = st.slider("Pick image", min_value=1, max_value=len(image_paths), value=1, step=1)

    with model_selector:
        # Select model
        model_option = st.selectbox(
            'Models',
            ('OCRModel', )
        )
        if model_option:

            weight_option = st.selectbox(
                'Model weights',
                os.listdir(f'assets/{model_option}')
            )

    # Show images
    ori_img, pred_img = st.columns([2, 3])
    if image_slider:
        img = Image.open(image_paths[image_slider-1])
        with ori_img:
            st.image(img, caption=image_paths[image_slider-1])
        with pred_img:
            if weight_option:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='jpeg')
                res = requests.post('http://localhost:8000/',
                                    files={'file': ("a.png", open(image_paths[image_slider-1], 'rb'))},
                                    data={
                                        'model': model_option,
                                        'weight': weight_option
                                    })
                if res.status_code == 200:
                    st.image(res.content, caption=image_paths[image_slider-1])


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main()