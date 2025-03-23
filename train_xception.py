import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import os
from tqdm import tqdm
import argparse
import random

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

def train_model(data_dir, num_epochs=100, batch_size=8, subset_size=None, cuda=True):
    """
    Train the Xception model on image dataset
    
    :param data_dir: Directory containing 'real' and 'fake' subdirectories
    :param num_epochs: Number of training epochs
    :param batch_size: Batch size for training
    :param subset_size: If provided, use only this many images for training
    :param cuda: Whether to use GPU
    """
    # Create dataset
    train_dataset = datasets.ImageFolder(
        data_dir,
        transform=xception_default_data_transforms['train']
    )
    
    # Create a subset if specified
    if subset_size is not None:
        total_size = len(train_dataset)
        if subset_size < total_size:
            indices = random.sample(range(total_size), subset_size)
            train_dataset = Subset(train_dataset, indices)
            print(f"Using {subset_size} images out of {total_size} total images")
    
    # Create data loader with larger batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    # Initialize model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    if cuda:
        model = model.cuda()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100 * correct/total:.2f}%'
            })
        
        pbar.close()
        
        print(f'Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}, '
              f'Accuracy: {100 * correct/total:.2f}%')
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                       help='Directory containing real and fake image folders')
    parser.add_argument('--num_epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--subset_size', '-s', type=int, default=None,
                       help='Number of images to use for training')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005,
                       help='Learning rate')
    parser.add_argument('--cuda', action='store_true',
                       help='Use GPU for training')
    parser.add_argument('--output_model', '-o', type=str, default='xception_df.pth',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Train the model
    model = train_model(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        cuda=args.cuda
    )
    
    # Save the trained model
    torch.save(model.state_dict(), args.output_model)
    print(f'Model saved to {args.output_model}') 