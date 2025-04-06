import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
from models.binarized_modules import BinarizeLinear


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        self.fc1 = BinarizeLinear(144, 384 * self.infl_ratio)
        self.htanh1 = torch.nn.Hardtanh()
        self.bn1 = torch.nn.BatchNorm1d(384 * self.infl_ratio)
        self.fc2 = BinarizeLinear(384 * self.infl_ratio, 384 * self.infl_ratio)
        self.htanh2 = torch.nn.Hardtanh()
        self.bn2 = torch.nn.BatchNorm1d(384 * self.infl_ratio)
        self.fc3 = BinarizeLinear(384 * self.infl_ratio, 384 * self.infl_ratio)
        self.htanh3 = torch.nn.Hardtanh()
        self.bn3 = torch.nn.BatchNorm1d(384 * self.infl_ratio)
        self.fc4 = torch.nn.Linear(384 * self.infl_ratio, 2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 12 * 12)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


def load_model(model_path='model.pth', use_cuda=torch.cuda.is_available()):
    """
    Load the trained model from the specified path

    Args:
        model_path (str): Path to the saved model
        use_cuda (bool): Whether to use CUDA if available

    Returns:
        model: The loaded neural network model
    """
    model = Net()

    if use_cuda:
        model.cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    return model


def preprocess_image(image_path):
    """
    Preprocess an image file for model inference

    Args:
        image_path (str): Path to the image file

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.39,), (0.28,))
    ])

    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(image)  # Add batch dimension
        img = transform(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def infer(model, input_tensor, use_cuda=torch.cuda.is_available()):
    """
    Run inference on input data

    Args:
        model: Trained neural network model
        input_tensor (torch.Tensor): Input data tensor
        use_cuda (bool): Whether to use CUDA if available

    Returns:
        tuple: (predicted_class, probabilities)
    """
    if use_cuda:
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        input_var = Variable(input_tensor)
        output = model(input_var)

        # Get probabilities
        probabilities = torch.exp(output)

        # Get predicted class
        pred_class = output.data.max(1, keepdim=True)[1].item()

        return pred_class, probabilities.cpu().numpy()[0]


def batch_inference(model, input_tensors, use_cuda=torch.cuda.is_available()):
    """
    Run inference on a batch of input data

    Args:
        model: Trained neural network model
        input_tensors (torch.Tensor): Batch of input data tensors
        use_cuda (bool): Whether to use CUDA if available

    Returns:
        tuple: (predicted_classes, probabilities)
    """
    if use_cuda:
        input_tensors = input_tensors.cuda()

    with torch.no_grad():
        input_var = Variable(input_tensors)
        output = model(input_var)

        # Get probabilities
        probabilities = torch.exp(output)

        # Get predicted classes
        pred_classes = output.data.max(1, keepdim=True)[1].squeeze().cpu().numpy()

        # Handle the case where there's only one sample
        if pred_classes.ndim == 0:
            pred_classes = np.array([pred_classes])

        return pred_classes, probabilities.cpu().numpy()


def process_image_folder(model, folder_path, batch_size=32, use_cuda=torch.cuda.is_available(), save_results=True,
                         output_file="results.csv"):
    """
    Process all images in a folder using the model

    Args:
        model: Trained neural network model
        folder_path (str): Path to the folder containing images
        batch_size (int): Batch size for processing
        use_cuda (bool): Whether to use CUDA if available
        save_results (bool): Whether to save results to a CSV file
        output_file (str): Path to save results if save_results is True

    Returns:
        dict: Dictionary mapping image filenames to their predictions and probabilities
    """
    folder = Path(folder_path)
    # Get all image files (common formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return {}

    results = {}
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

        # Prepare batch
        batch_tensors = []
        valid_files = []

        for img_file in batch_files:
            tensor = preprocess_image(img_file)
            if tensor is not None:
                batch_tensors.append(tensor)
                valid_files.append(img_file)

        if not batch_tensors:
            print(f"Batch {batch_idx + 1}/{num_batches}: No valid images found")
            continue

        # Concatenate tensors
        batch_input = torch.cat(batch_tensors, 0)

        # Run inference
        pred_classes, probs = batch_inference(model, batch_input, use_cuda)

        # Store results
        for i, (img_file, cls, prob) in enumerate(zip(valid_files, pred_classes, probs)):
            results[img_file.name] = {
                'file_path': str(img_file),
                'predicted_class': int(cls),
                'probabilities': prob.tolist()
            }

        print(f"Processed batch {batch_idx + 1}/{num_batches} ({len(valid_files)} images)")

    # Save results to CSV if requested
    if save_results:
        import csv
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'file_path', 'predicted_class', 'prob_class_0', 'prob_class_1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            false_count = 0
            for filename, data in results.items():
                writer.writerow({
                    'filename': filename,
                    'file_path': data['file_path'],
                    'predicted_class': data['predicted_class'],
                    'prob_class_0': data['probabilities'][0],
                    'prob_class_1': data['probabilities'][1]
                })
                if data['predicted_class'] != 1:
                    false_count += 1

        print(f"Results saved to {output_file}")
        print(f"Accuracy: {false_count / len(results) }")

    return results


def main():
    """
    Example usage with image folder processing
    """
    import argparse

    parser = argparse.ArgumentParser(description='CAN Model Inference')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to the trained model')
    parser.add_argument('--image-folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file path')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(f"Loading model from {args.model}")
    model = load_model(args.model, use_cuda)

    print(f"Processing images in {args.image_folder}")
    results = process_image_folder(
        model,
        args.image_folder,
        batch_size=args.batch_size,
        use_cuda=use_cuda,
        output_file=args.output
    )

    print(f"Processed {len(results)} images")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()