import argparse
import torch
from torchvision import models
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def get_input_args():
    """
    Parse command-line arguments for prediction.
    """
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
    
    # path to image & checkpoint
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the saved checkpoint.pth file.')
    
    top K classes
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K')
    
    # Optional: category names JSON
    parser.add_argument('--category_names', type=str,
                        help='Path to a JSON file mapping labels to flower names.')
    
    # use GPU
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Load a saved checkpoint and rebuild the model.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Check architecture
    arch = checkpoint['arch']
    if arch == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = checkpoint['classifier']
    elif arch == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        model.classifier = checkpoint['classifier']
    else:
        raise ValueError(f"Unsupported architecture {arch}")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a tensor ready for inference.
    """
    img = Image.open(image_path)
    
    # Define transforms to match training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    img = transform(img)
    return img

def predict(image_path, model, device, top_k=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    model.to(device)
    
    # Process the image
    img_tensor = process_image(image_path).unsqueeze_(0).to(device)
    
    with torch.no_grad():
        logps = model(img_tensor)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
    
    # Convert to lists
    top_p = top_p[0].tolist()
    top_class = top_class[0].tolist()
    
    # Invert the class_to_idx dictionary
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[c] for c in top_class]
    
    return top_p, top_labels

def main():
    args = get_input_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Make prediction
    top_p, top_labels = predict(args.image_path, model, device, top_k=args.top_k)
    
    # If category names provided, map labels to actual flower names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        # Convert numeric labels to flower names
        top_flowers = [cat_to_name[str(lbl)] for lbl in top_labels]
    else:
        top_flowers = top_labels
    
    # Print results
    print("Top probabilities:", top_p)
    print("Top labels:", top_labels)
    print("Flower names:" if args.category_names else "Labels:", top_flowers)

if __name__ == '__main__':
    main()
