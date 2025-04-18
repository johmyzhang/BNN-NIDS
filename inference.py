import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from cicids_dataset import NetworkFeatureDataset
from utils.binarized_modules import BinarizeLinear, CoarseNormalization
import numpy as np

torch.set_printoptions(precision=4)
# Import your model definition.
# Here we assume your model class Net is defined in train.py.
class Net(nn.Module):
    def __init__(self, fine_norm, trace):
        super(Net, self).__init__()
        self.infl_ratio = 4
        self.trace = trace
        if fine_norm:
            print('Using Layer Normalization')
            # Flattened input size is 81 (9x9)
            self.fc1 = BinarizeLinear(16, 16 * self.infl_ratio, bias=False)
            self.bn1 = nn.LayerNorm(16 * self.infl_ratio, elementwise_affine=False, bias=False)
            self.htanh1 = nn.Hardtanh()

            self.fc2 = BinarizeLinear(16 * self.infl_ratio, 16 * self.infl_ratio, bias=False)
            self.bn2 = nn.LayerNorm(16 * self.infl_ratio, elementwise_affine=False, bias=False)
            self.htanh2 = nn.Hardtanh()

            self.fc3 = BinarizeLinear(16 * self.infl_ratio, 16 * self.infl_ratio, bias=False)
            self.bn3 = nn.LayerNorm(16 * self.infl_ratio, elementwise_affine=False, bias=False)
            self.htanh3 = nn.Hardtanh()

            self.fc4 = BinarizeLinear(16 * self.infl_ratio, 2, bias=False)
        else:
            print('Using Coarse Normalization')
            self.fc1 = BinarizeLinear(16, 16 * self.infl_ratio, bias=False)
            self.bn1 = CoarseNormalization(16 * self.infl_ratio, elementwise_affine=False, bias=False)
            self.htanh1 = nn.Hardtanh()

            self.fc2 = BinarizeLinear(16 * self.infl_ratio, 16 * self.infl_ratio, bias=False)
            self.bn2 = CoarseNormalization(16 * self.infl_ratio, elementwise_affine=False, bias=False)
            self.htanh2 = nn.Hardtanh()

            self.fc3 = BinarizeLinear(16 * self.infl_ratio, 16 * self.infl_ratio, bias=False)
            self.bn3 = CoarseNormalization(16 * self.infl_ratio, elementwise_affine=False, bias=False)
            self.htanh3 = nn.Hardtanh()

            self.fc4 = BinarizeLinear(16 * self.infl_ratio, 2, bias=False)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 16)

        # First block
        if self.trace:
            print('Input layer in--------------')
            print(x)
        x = self.fc1(x)
        if self.trace:
            print('Input layer linear transformation out--------------')
            print(x)
        x = self.bn1(x)
        if self.trace:
            print('Input layer normalization out--------------')
            print(x)
        x = self.htanh1(x)

        # Second block
        if self.trace:
            print('Hidden 1 layer in--------------')
            print(x)
        x = self.fc2(x)
        if self.trace:
            print('Hidden 1 layer linear transformation out--------------')
            print(x)
        x = self.bn2(x)
        if self.trace:
            print('Hidden 1 layer normalization out--------------')
            print(x)
        x = self.htanh2(x)
        # Third block
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh3(x)

        # Output layer
        x = self.fc4(x)
        return torch.log_softmax(x, dim=1)  # More efficient than nn.LogSoftmax



def load_model(model_path, device, fine_norm, trace):
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

    model = Net(fine_norm, trace)  # Make sure this architecture matches your training code.
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode.
    return model


def main():
    parser = argparse.ArgumentParser(description='Batch inference on images.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (pth)')
    parser.add_argument('--test-datapath', type=str, required=True, help='Folder containing images for inference')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--coarse-norm', type=bool, default=False, help='Batch size for inference')
    parser.add_argument('--trace', type=bool, default=False, help='Batch size for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fine_norm = not args.coarse_norm

    # Load the model.
    model = load_model(args.model, device, fine_norm, args.trace)

    test_data = NetworkFeatureDataset(
        csv=args.test_datapath,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Inference accuracy: {correct / len(test_data)}")


if __name__ == '__main__':
    main()
