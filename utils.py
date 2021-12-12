import streamlit as st
from typing import List
from glob import glob
import json
from models import model
import torch

@st.cache
def load_image_list() -> List[str]:
    return glob("./assets/images/*")

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_weight(model_name, weight):
    with open('assets/config.json') as f:
        config = json.load(f)
    loaded_model = getattr(model, model_name)(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f'./assets/{model_name}/{weight}', map_location=device)
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.eval()
    return loaded_model