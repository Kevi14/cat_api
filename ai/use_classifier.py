import torch
import torchvision.transforms as transforms
from .model import Net
from PIL import Image
import pandas as pd

def find_cat_breed(image_file):
    PATH = 'ai_net.pth'
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    if not model == 'cpu':
        model.cuda()

    image = Image.open(image_file)

    # Convert image to RGB format
    image = image.convert('RGB')

    # Resize image to 224x224
    image = image.resize((224, 224))

    image_converter = transforms.ToTensor()

    image_tensor = image_converter(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    scores = model(image_tensor.to(device=model.device))

    _, predictions = scores.max(1)

    # breed_name = names_df[names_df["value"] == int(predictions.item())]["name"].values[0]
    # print(breed_name)
    return int(predictions.item())
