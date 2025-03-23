import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import os
from tqdm import tqdm
import argparse
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

def evaluate_model(model, data_loader, device, phase='val'):
    """
    Evaluate the model on validation/test set
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'{phase.capitalize()} Phase')
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

def train_model(train_dir, val_dir, test_dir, num_epochs=100, batch_size=8, learning_rate=0.00005, subset_size=None):
    """
    Train and evaluate the Xception model
    :param subset_size: If provided, use only this many images for each split (train/val/test)
    """
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() 
                        else "cuda" if torch.cuda.is_available() 
                        else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=xception_default_data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=xception_default_data_transforms['test'])
    test_dataset = datasets.ImageFolder(test_dir, transform=xception_default_data_transforms['test'])
    
    # Create subsets if specified
    if subset_size is not None:
        for dataset, name in [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]:
            total_size = len(dataset)
            if subset_size < total_size:
                indices = random.sample(range(total_size), subset_size)
                if name == 'train':
                    train_dataset = Subset(dataset, indices)
                elif name == 'val':
                    val_dataset = Subset(dataset, indices)
                else:
                    test_dataset = Subset(dataset, indices)
                print(f"Using {subset_size} images out of {total_size} total images for {name}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    
    print(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'train_loss': f'{running_loss/len(pbar):.4f}'})
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, 'val')
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Val Precision: {val_metrics["precision"]:.4f}')
        print(f'Val Recall: {val_metrics["recall"]:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}')
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model for testing
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, device, 'test')
    
    print('\nFinal Test Results:')
    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Test Precision: {test_metrics["precision"]:.4f}')
    print(f'Test Recall: {test_metrics["recall"]:.4f}')
    print(f'Test F1: {test_metrics["f1"]:.4f}')
    
    return model, test_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dir', '-d', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--val_dir', '-v', type=str, required=True,
                       help='Directory containing validation images')
    parser.add_argument('--test_dir', '-t', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--num_epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005)
    parser.add_argument('--subset_size', '-s', type=int, default=None,
                       help='Number of images to use for each split (train/val/test)')
    parser.add_argument('--output_model', '-o', type=str, default='xception_df.pth')
    
    args = parser.parse_args()
    
    # Train and evaluate the model
    model, test_metrics = train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size
    )
    
    # Save the best model
    torch.save(model.state_dict(), args.output_model)
    print(f'Model saved to {args.output_model}') 