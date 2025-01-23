import torch
from torchvision import datasets, transforms


def calculate_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)

    mean, std = calculate_mean_std(train_dataset)
    print(f'Mean: {mean}')
    print(f'Std: {std}')