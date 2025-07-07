import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import preProcessing as pre
import gradio as gr


model = efficientnet_v2_s()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)

checkpoint = torch.load('../savings/best_model.pt', map_location='cpu')

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

def predict(img: Image.Image):
    img_t = pre.test_transform(img).unsqueeze(0) 
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs[0], dim=0)
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


gr.Interface(fn=predict,
             theme=gr.themes.Ocean(),
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             example=["cocacola.jpeg", "bottiglia.jpg", "carta.jpg", "vetro.png"]).launch()