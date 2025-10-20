import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from models.simple_cnn import SimpleCNN

def save_prediction_grid(images, labels, preds, classes, out_path, mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)):
    images = images.cpu()
    n = min(8, images.size(0))
    fig = plt.figure(figsize=(12,3))
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        img = images[i].numpy().transpose((1,2,0))
        img = (img * np.array(std)) + np.array(mean)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'T:{classes[labels[i]]}\nP:{classes[preds[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    os.makedirs('results', exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    classes = testset.classes

    # Load best model
    ckpt = os.path.join('saved_models', 'cnn_best.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}. Run train.py first.')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f'Test accuracy: {acc*100:.2f}%')

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # save sample predictions
    dataiter = iter(testloader)
    imgs, labels = next(dataiter)
    with torch.no_grad():
        outputs = model(imgs.to(device))
        _, preds = outputs.max(1)
    save_prediction_grid(imgs, labels.numpy(), preds.cpu().numpy(), classes, 'results/sample_predictions.png')
    print('Saved results to results/')

if __name__ == '__main__':
    main()
