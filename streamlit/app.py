import requests
from PIL import Image

import streamlit as st

st.set_page_config(page_title="图片转换到Latex")

st.title("图片转换到Latex页面")


uploaded_file = st.file_uploader(
    "请上传一张包含latext的图片",
    type=["png", "jpg"],
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)


if st.button("Convert"):
    if uploaded_file is not None and image is not None:
        files = {"file": uploaded_file.getvalue()}
        with st.spinner("开始转换..."):
            response = requests.post("http://0.0.0.0:7800/predict/", files=files)
        latex_code = response.json()["data"]["pred"]
        st.code(latex_code)
        st.markdown(f"${latex_code}$")
    else:
        st.error("Please upload an image.")
