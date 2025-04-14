from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import time
from cicids_dataset import NetworkFeatureDataset
from models.binarized_modules import BinarizeLinear

# Training settings
parser = argparse.ArgumentParser(description='CAN Data Test')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-datapath', type=str, default='train_data')
parser.add_argument('--test-datapath', type=str, default='test_data')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use automatic mixed precision')
parser.add_argument('--prefetch-factor', type=int, default=2,
                    help='number of batches to prefetch')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--compile', action='store_true', default=False,
                    help='use torch.compile for faster execution')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Data loading parameters
kwargs = {
    'num_workers': args.num_workers,
    'pin_memory': True,
    'prefetch_factor': args.prefetch_factor
} if args.cuda else {}

def get_datasets():
    # Data augmentation for training
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])
    #
    # # No augmentation for testing
    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])
    #
    # training_data = PacketImageDataset(
    #     img_dir=args.train_datapath,
    #     annotations_file='train_ddos.csv',
    #     transform=train_transform)
    #
    # test_data = PacketImageDataset(
    #     img_dir=args.test_datapath,
    #     annotations_file='test_ddos.csv',
    #     transform=test_transform)

    training_data = NetworkFeatureDataset(
        csv=args.train_datapath,
    )

    test_data = NetworkFeatureDataset(
        csv=args.test_datapath,
    )

    return training_data, test_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        # Flattened input size is 81 (9x9)
        self.fc1 = BinarizeLinear(16, 16 * self.infl_ratio)
        self.bn1 = nn.BatchNorm1d(16 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()

        self.fc2 = BinarizeLinear(16 * self.infl_ratio, 16 * self.infl_ratio)
        self.bn2 = nn.BatchNorm1d(16 * self.infl_ratio)
        self.htanh2 = nn.Hardtanh()

        self.fc3 = BinarizeLinear(16 * self.infl_ratio, 16 * self.infl_ratio)
        self.bn3 = nn.BatchNorm1d(16 * self.infl_ratio)
        self.htanh3 = nn.Hardtanh()

        self.drop = nn.Dropout(0.5)
        self.fc4 = nn.Linear(16 * self.infl_ratio, 2)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 16)

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


def train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device):
    model.train()
    start_time = time.time()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()

        # Use AMP for mixed precision training
        if args.amp:
            with amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)

            # Scale the loss and call backward
            scaler.scale(loss).backward()

            # Apply binary constraints
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)

            # Update parameters with gradient scaling
            scaler.step(optimizer)
            scaler.update()

            # Apply binarization constraint after optimizer step
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
        else:
            # Regular precision training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Apply binary constraints
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)

            optimizer.step()

            # Apply binarization constraint after optimizer step
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds. Avg loss: {running_loss / len(train_loader):.6f}")

    return running_loss / len(train_loader)


def validate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy


def main():
    # Get device
    device = torch.device("cuda" if args.cuda else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load datasets
    training_data, test_data = get_datasets()

    # Create data loaders
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )

    # Create model
    model = Net().to(device)

    # Use torch.compile for faster execution (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True
    )

    # Initialize gradient scaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # Training loop
    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device)

        # Evaluate on test set
        test_loss, accuracy = validate(model, test_loader, criterion, device)

        # Update learning rate based on validation performance
        scheduler.step(test_loss)

        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, "best_model.pth")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, f"checkpoint_epoch_{epoch}.pth")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()