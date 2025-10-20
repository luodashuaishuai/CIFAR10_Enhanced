import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from models.simple_cnn import SimpleCNN


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Datasets and loaders
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)  # 改为0避免多进程问题

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)  # 改为0避免多进程问题

    classes = trainset.classes

    # Model, loss, optimizer, scheduler
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, loss):
            if self.best_loss is None:
                self.best_loss = loss
                return
            if loss < self.best_loss - self.min_delta:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    early_stopper = EarlyStopping(patience=7)

    num_epochs = 100
    best_val_acc = 0.0
    train_losses, val_losses = [], []

    os.makedirs('saved_models', exist_ok=True)
    save_path = os.path.join('saved_models', 'cnn_best.pth')

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)

        # validation on test set (as val)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)

        print(
            f'Epoch {epoch:03d} | Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'  Saved best model (val_acc={best_val_acc:.4f})')

        # step scheduler
        scheduler.step()

        # early stopping check
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f'Early stopping at epoch {epoch}')
            break

    # Plot loss curves
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curve.png', bbox_inches='tight')
    plt.close()
    print('Saved loss curve to results/loss_curve.png')


if __name__ == '__main__':
    main()