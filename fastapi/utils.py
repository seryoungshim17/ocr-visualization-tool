import json
import json
import torch
from models.ocrmodel import OCRModel
def load_weight(model_name, weight):
    with open('assets/config.json') as f:
        config = json.load(f)
    config['recognition']['name'] = model_name
    loaded_model = OCRModel(config)
    
    device = torch.device("cpu")
    checkpoint = torch.load(f'./assets/{model_name}/{weight}', map_location=device)
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.eval()
    return loaded_model