import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
import argparse
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on test set
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Testing')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    metrics = {
        'loss': running_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def test_model(test_dir, weights_path, batch_size=8, subset_size=None):
    """
    Test the Xception model using pre-trained weights
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=xception_default_data_transforms['test'])
    
    # Create subset if specified
    if subset_size is not None:
        total_size = len(test_dataset)
        if subset_size < total_size:
            indices = random.sample(range(total_size), subset_size)
            test_dataset = Subset(test_dataset, indices)
            print(f"Using {subset_size} images out of {total_size} total images for testing")
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Test Dataset size: {len(test_dataset)}")

    # Initialize model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    
    # Load weights
    print(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Test the model
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print('\nTest Results:')
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    return test_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_dir', '-t', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--weights_path', '-w', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--subset_size', '-s', type=int, default=None,
                       help='Number of images to use for testing')
    
    args = parser.parse_args()
    
    # Test the model
    test_metrics = test_model(
        test_dir=args.test_dir,
        weights_path=args.weights_path,
        batch_size=args.batch_size,
        subset_size=args.subset_size
    ) 