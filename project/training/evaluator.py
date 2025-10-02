import torch
import warnings

def evaluate_loop(model, dataloader, criterion, device):
    """Evaluates the model on a dataloader that wraps an IterableDataset."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    # Suppress the UserWarning from PyTorch about the length of the IterableDataset
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Length of IterableDataset .* was reported to be .* but .* samples have been fetched.",
            category=UserWarning
        )
        with torch.no_grad():
            for features, labels, weights in dataloader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item() * features.size(0)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
    if total_samples == 0:
        return 0.0, 0.0, [], []
        
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy, all_labels, all_preds