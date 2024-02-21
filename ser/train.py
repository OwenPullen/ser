import torch
# from ser.data import training_dataloader, validation_dataloader
import torch.nn.functional as F
from ser.logging import log_training, log_validation
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def train(model, device, optimizer, epoch, training_dataloader, logging_dir):
    # train
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            batch = f"{i}/{len(training_dataloader)}"
            print(
                f"Train Epoch: {epoch} | Batch: {batch}"
                f"| Loss: {loss.item():.4f}"
            )
            log_training(epoch, batch, loss.item(), logging_dir)
        
        

def validate(model, device, validation_dataloader, epoch, logging_dir):
        # validate
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                model.eval()
                output = model(images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )
            log_validation(epoch, val_loss, val_acc, logging_dir)

            
            
def training_wrapper(model, device, optimizer,
                      epochs, training_dataloader,
                        validation_dataloader, logging_dir):
    for epoch in range(epochs):
          train(model, device, optimizer, epoch, training_dataloader, logging_dir)         
          validate(model, device, validation_dataloader, epoch, logging_dir)
    
    torch.save(model.state_dict(), PROJECT_ROOT / "experiments" / logging_dir / "trained_model.pth")
          