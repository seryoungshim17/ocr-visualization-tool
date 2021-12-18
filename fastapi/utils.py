import json
import json
import torch
def load_weight(model_name, weight):
    with open('assets/config.json') as f:
        config = json.load(f)
    loaded_model = getattr(__import__(f"models.{model_name.lower()}", fromlist=(model_name)), model_name)(config)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    checkpoint = torch.load(f'./assets/{model_name}/{weight}', map_location=device)
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.eval()
    return loaded_model