import argparse
import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from models.binarized_modules import BinarizeLinear
import numpy as np

# Import your model definition.
# Here we assume your model class Net is defined in main_can.py.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        # Flattened input size is 81 (9x9)
        self.fc1 = BinarizeLinear(81, 128 * self.infl_ratio)
        self.bn1 = nn.BatchNorm1d(128 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()

        self.fc2 = BinarizeLinear(128 * self.infl_ratio, 128 * self.infl_ratio)
        self.bn2 = nn.BatchNorm1d(128 * self.infl_ratio)
        self.htanh2 = nn.Hardtanh()

        self.fc3 = BinarizeLinear(128 * self.infl_ratio, 128 * self.infl_ratio)
        self.bn3 = nn.BatchNorm1d(128 * self.infl_ratio)
        self.htanh3 = nn.Hardtanh()

        self.drop = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128 * self.infl_ratio, 5)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 9 * 9)

        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)

        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)

        # Third block
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.drop(x)

        # Output layer
        x = self.fc4(x)
        return torch.log_softmax(x, dim=1)  # More efficient than nn.LogSoftmax



def load_model(model_path, device):
    """
    Load the model from the checkpoint file and remove any unwanted key prefixes.
    """
    checkpoint = torch.load(model_path, map_location=device)
    # Extract the model state dict. Change the key if needed.
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Remove unwanted prefix (e.g., '_orig_mod.') from each key.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    model = Net()  # Make sure this architecture matches your training code.
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode.
    return model


def load_images_from_folder(folder, transform):
    """
    Load images from the specified folder using the provided transform.
    Returns a list of filenames and a list of processed images.
    """
    filenames = []
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(folder, file)
            try:
                image = Image.open(file_path).convert('L')
                img1d = np.array(image)
                img1d_ = transform(img1d)
                images.append(img1d_)
                filenames.append(file)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
    return filenames, images


def main():
    parser = argparse.ArgumentParser(description='Batch inference on images.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (pth)')
    parser.add_argument('--image-folder', type=str, required=True, help='Folder containing images for inference')
    parser.add_argument('--output-csv', type=str, default='inference_results.csv', help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transforms (adjust size and normalization as required by your model).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.25,), (0.35,))
        # If your model expects normalized images, uncomment and adjust the following:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the model.
    model = load_model(args.model, device)

    # Load images from the specified folder.
    filenames, images = load_images_from_folder(args.image_folder, transform)
    if not images:
        print("No images found in the folder.")
        return

    results = []
    num_images = len(images)
    with torch.no_grad():
        # Process images in batches.
        for i in range(0, num_images, args.batch_size):
            batch_images = images[i:i + args.batch_size]
            batch_filenames = filenames[i:i + args.batch_size]
            batch_tensor = torch.stack(batch_images).to(device)

            outputs = model(batch_tensor)
            # Convert model outputs to probabilities.
            probabilities = F.softmax(outputs, dim=1)

            for j in range(len(batch_images)):
                probs = probabilities[j].cpu().numpy().tolist()
                # Predicted class: index with highest probability.
                pred_class = probs.index(max(probs))
                results.append({
                    'filename': batch_filenames[j],
                    'probabilities': probs,
                    'result': pred_class
                })

    # Write inference results to a CSV file.
    with open(args.output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['filename', 'probabilities', 'result']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            # Convert the list of probabilities to a comma-separated string.
            res['probabilities'] = ','.join([f'{p:.4f}' for p in res['probabilities']])
            writer.writerow(res)

    print(f"Inference results saved to {args.output_csv}")


if __name__ == '__main__':
    main()
