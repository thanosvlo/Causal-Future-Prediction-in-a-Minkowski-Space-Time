from torch.utils.data import Dataset
import os
import os.path
import numpy as np
import torchvision
import torch


class MovingMNIST(Dataset):
    def __init__(self, opt, root, train=True, labels=False):
        self.opt = opt
        self.root = root
        self.labels = labels
        if train:
            data = np.load(os.path.join(self.root,'moving_mnist_train_with_labels.npz'))
            data = data['arr_0']
            data = np.reshape(data, [-1, 30, 1, 64, 64])
            self.data = torch.from_numpy(data)
        else:
            data = np.load(os.path.join(self.root, 'moving_mnist_test_with_labels.npz'))
            data = data['arr_0']
            data = np.reshape(data, [-1, 30, 1, 64, 64])
            self.data = torch.from_numpy(data)

    def __getitem__(self, index):

        image = self.data[index]
        image = image/255.

        return image

    def __len__(self):
        return len(self.data)




def get_moving_mnist_dataloaders(opt):
    """
        creates and returns the Moving MNIST dataset and dataloaders,
        either with train/val split, or train+val/test split
        :param opt:
        :return: train_loader, train_dataset,
        test_loader, test_dataset - corresponds to validation or test set depending on opt.validate
        """

    num_workers = 16

    train_dataset = MovingMNIST(opt, opt.data_input_dir, train=True)
    test_dataset = MovingMNIST(opt, opt.data_input_dir, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset
