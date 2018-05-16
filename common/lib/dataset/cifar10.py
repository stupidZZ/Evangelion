import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Cifar10(data.Dataset):
    def __init__(self, image_set, root_path, data_path, scales):
        super(Cifar10, self).__init__()
        self.data_path = data_path
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.image_set = image_set

        assert len(scales) == 1

        if os.path.basename(data_path) != "cifar-10-batches-py":
            raise "Data_path must be 'cifar-10-batches-py' "

        if self.image_set == "cifar10_train":
            transform = transforms.Compose([
                transforms.Resize(scales[0]),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.dataset = datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
        elif self.image_set == "cifar10_test":
            transform = transforms.Compose([
                transforms.Resize(scales[0]),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.dataset = datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.__len__()