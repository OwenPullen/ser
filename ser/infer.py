from ser.data import test_dataloader, normalize
from ser.transforms import transforms
import torch

def load_data():
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        mages, labels = next(iter(dataloader))

@torch.no_grad()
def test_model_inference(params, model, image, label):
    print(f"Runing inference on model - {params.name}")
    print(f"Hyperparameters:\n",
          f"EPOCHS: {params.epochs}\n",
          f"LEARNING RATE: {params.lr}\n",
          f"Label of image: {label}"
        )
    model.eval()
    output = model(image)
    prediction = torch.argmax(output, 1, keepdim=True).item()
    confidence = max(list(torch.exp(output).numpy()[0]))
    
    pixels = image[0][0]
    print(generate_ascii_art(pixels))

def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)

def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "    