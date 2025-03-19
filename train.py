import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os
import time
from collections import OrderedDict

def get_input_args():
    """
    Defines command-line arguments and returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train  and save to checkpoint.")
    
    
    parser.add_argument('data_dir', type=str, help='Path to the folder of images')
    
 
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save checkpoints (default: current folder)')
    # choose model
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='Model architecture. Choose between "densenet121" or "vgg16" (default: densenet121).')
    
    
    # hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units in the classifier (default: 512)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    
   
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    
    return parser.parse_args()

def main():
     # 1. input args
    print("Script starting...", flush=True)

    args = get_input_args()
    print("Starting training with these parameters:", args, flush=True)
    
    start_time = time.time()
    
    # 2. Set up GPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 3. Load and transform data
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # Define transforms
    train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

  
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transforms = valid_transforms  # often same as valid
    
    # Create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    # 4. Build model
    if args.arch == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        in_features = model.classifier.in_features
    elif args.arch == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        in_features = model.classifier[0].in_features
    else:
        print(f"Architecture {args.arch} is not supported. Exiting.")
        return

    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Create custom classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(args.hidden_units, 102)),  # 102 classes for flower dataset
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.to(device)
    
    # 5. Set up loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # 6. Train model
    epochs = args.epochs
    steps = 0
    print_every = 2  # prints every 2 batches
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        print(f"Epoch {epoch+1}/{epochs} starting...", flush=True)
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print after every 2 batches
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for val_inputs, val_labels in validloader:
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        logps_val = model(val_inputs)
                        batch_loss = criterion(logps_val, val_labels)
                        val_loss += batch_loss.item()
                        
                        ps_val = torch.exp(logps_val)
                        top_p, top_class = ps_val.topk(1, dim=1)
                        equals = top_class == val_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}", flush=True)
                running_loss = 0
                model.train()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total Training Time: {minutes}m {seconds}s", flush=True)
    
    # 7. Test model
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test Loss: {test_loss/len(testloader):.3f}.. "
          f"Test Accuracy: {accuracy/len(testloader):.3f}", flush=True)
    
   # 8. Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {
        'arch': args.arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    
    save_path = os.path.join(args.save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()