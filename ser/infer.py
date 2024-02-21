from ser.data import test_dataloader, normalize
from ser.transforms import transforms
import torch

def load_data():
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
        while labels[0].item() != label:
                images, labels = next(iter(dataloader))
    
def test_model_inference(model, device , run_path, label):
     # select image to run inference for

    
        # run inference
    with torch.no_grad():
        for data in dataloader:
            image, labels = data
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print(f"Predicted: {predicted.item()}")
            print(f"Actual: {labels.item()}")
            return images, labels, predicted