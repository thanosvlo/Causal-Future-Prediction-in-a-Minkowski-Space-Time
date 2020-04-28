import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
import os
import numpy as np
import math
from numpy import prod
from .vae import VAE
from pvae.utils import Constants


from pvae.distributions import RiemannianNormal, WrappedNormal
from torch.distributions import Normal

from pvae import manifolds
from .architectures import EncLinear, DecLinear, EncWrapped, DecWrapped, EncMob, DecMob, DecGeo, DecBernouilliWrapper,\
    EncWrapped_Conv,DecWrapped_Conv

data_size = torch.Size([1, 32, 32])


class Mnist(VAE):
    def __init__(self, params):
        self.params = params
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        super(Mnist, self).__init__(
            eval(params.prior),   # prior distribution
            eval(params.posterior),   # posterior distribution
            dist.RelaxedBernoulli,        # likelihood distribution
            eval('Enc' + params.enc)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
            DecBernouilliWrapper(eval('Dec' + params.dec)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim)),
            params
        )
        self.manifold = manifold
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Mnist'

    def init_last_layer_bias(self, train_loader):
        if not hasattr(self.dec.dec.fc31, 'bias'): return
        with torch.no_grad():
            p = torch.zeros(prod(data_size[1:]), device=self._pz_mu.device)
            N = 0
            for i, (data, _) in enumerate(train_loader):
                if 'info' in self.params.obj:
                    data = data[:,0,...]
                data = data.to(self._pz_mu.device)
                B = data.size(0)
                N += B
                p += data.view(-1, prod(data_size[1:])).sum(0)
            p /= N
            p += 1e-4
            self.dec.dec.fc31.bias.set_(p.log() - (1 - p).log())

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        # this is required if using the relaxedBernoulli because it doesn't
        # handle scoring values that are actually 0. or 1.
        tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda p: p.clamp(Constants.eta, 1 - Constants.eta))
        ])
        train_loader = DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def _get_random_number(self,pos_index,labels,max):
        found = False
        rnd_num = None
        while not found:
            rnd_num = np.random.randint(0,max)
            if labels[rnd_num,0,0] != labels[pos_index,0,0]:
                found = True
        return rnd_num

    def _process_data(self, data_filename, label_filename,flag):
        labels = np.load(label_filename)
        if 'npz' in os.path.basename(label_filename):
            labels = labels['arr_0']
            labels = labels[:,0]
        if 'info' in self.params.obj and not flag:
            true_labels = np.array([labels[i:i + 30] for i in range(0, len(labels), 30)])
            labels = np.array([np.concatenate((np.ones(self.params.pos_samples), np.zeros(self.params.neg_samples)),
                                              axis=0) for i in range(true_labels.shape[0])])
        if 'only' in self.params.runId:
            lbls_to_get = self.params.runId.split('_')
            lbl_1 = int(lbls_to_get[-1])
            lbl_2 = int(lbls_to_get[-2])
            temp_indx_1 = np.where(labels==lbl_1)[0]
            temp_indx_2 = np.where(labels==lbl_2)[0]
            temp_indx = np.concatenate((temp_indx_1,temp_indx_2))
            labels = labels[temp_indx]

        labels = torch.Tensor(labels)

        data = np.load(data_filename)

        if 'npz' in os.path.basename(label_filename):
            if '28' in self.params.runId:
                data = data['arr_0'].reshape(-1, 1, 28, 28)
            elif '32' in self.params.runId:
                data = data['arr_0'].reshape(-1, 1, 32, 32)
            else:
                data = data['arr_0'].reshape(-1, 1, 64, 64)

        data = data / np.float32(255)
        data = np.transpose(data,[0,1,3,2])
        if 'only' in self.params.runId:
            data = data[temp_indx]

        if 'info' in self.params.obj and not flag:
            data = np.array([data[i:i + 30] for i in range(0, len(data), 30)])
            _data = []
            for i in range(data.shape[0]):
                rnd_num = np.random.randint(0, 30 - max(self.params.pos_samples, self.params.neg_samples))
                temp_positives = data[i, rnd_num:rnd_num+self.params.pos_samples, ...]
                j = self._get_random_number(i,true_labels,data.shape[0])
                temp_negatives = data[j,rnd_num:rnd_num+self.params.neg_samples,...]
                _data.append(np.concatenate([temp_positives,temp_negatives], axis=0))
            data = torch.Tensor(_data)
        else:
            data = torch.Tensor(data)


        return data, labels


    def getDataLoaders_moving_mnist(self, batch_size, shuffle=True, device="cuda",args=None, transform=True):

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        flag = os.path.exists(os.path.join(args.data_path, 'pvae_info_train_data.npy'))

        if flag and 'info' in self.params.obj:
            data_filename_train = os.path.join(args.data_path, 'pvae_info_train_data.npy')
            label_filename_train = os.path.join(args.data_path, 'pvae_info_train_data_labels.npy')

            data_filename_test = os.path.join(args.data_path, 'pvae_info_test_data.npy')
            label_filename_test = os.path.join(args.data_path, 'pvae_info_test_data_labels.npy')
        elif '28' in self.params.runId:
            data_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_28x28.npz')
            label_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_28x28_labels.npz')

            data_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_28x28.npz')
            label_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_28x28_labels.npz')
        elif '18_pixels' in self.params.runId:
            data_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_32x32_18pixels.npz')
            label_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_32x32_18pixels_labels.npz')

            data_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_32x32_18pixels.npz')
            label_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_32x32_18pixels_labels.npz')

        elif '32' in self.params.runId:
            data_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_32x32.npz')
            label_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_32x32_labels.npz')

            data_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_32x32.npz')
            label_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_32x32_labels.npz')

        else:
            data_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels.npz')
            label_filename_train = os.path.join(args.data_path, 'moving_mnist_train_with_labels_labels.npz')

            data_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels.npz')
            label_filename_test = os.path.join(args.data_path, 'moving_mnist_test_with_labels_labels.npz')

        train_data, train_labels = self._process_data(data_filename_train,label_filename_train,flag)
        test_data, test_labels = self._process_data(data_filename_test,label_filename_test,flag)

        # train_data = torch.nn.functional.interpolate(train_data, 28)
        # test_data = torch.nn.functional.interpolate(test_data, 28)
        train_dataset = torch.utils.data.TensorDataset(train_data,train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)

        train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle, **kwargs)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader


    def sample_new(self,N):
        samples = super(Mnist, self).sample_new(N)
        return samples

    def generate(self, runPath, epoch):
        N, K = 64, 9
        mean, means, samples = super(Mnist, self).generate(N, K)
        save_image(mean.data.cpu(), '{}/gen_mean_{:03d}.png'.format(runPath, epoch))
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Mnist, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))



class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)