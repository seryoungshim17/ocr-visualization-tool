import streamlit as st
from typing import List
from glob import glob

@st.cache
def load_image_list() -> List[str]:
    return glob("./assets/images/*")