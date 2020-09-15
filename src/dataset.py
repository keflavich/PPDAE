import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
# from sklearn import preprocessing

root = os.getcwd()
colab_root = '/content/drive/My Drive'
exalearn_root = '/home/jorgemarpa/data/imgs'


class MyRotationTransform:
    """Rotate by a random N times 90 deg."""

    def __init__(self):
        pass

    def __call__(self, x):
        shape = x.shape
        return np.rot90(x, np.random.choice([0, 1, 2, 3]),
                        axes=[-2, -1]).copy()


class MyFlipVerticalTransform:
    """Random vertical flip."""

    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, x):
        if np.random.uniform() >= self.prob:
            return np.flip(x, -2).copy()
        else:
            return x


class MyNormTransform:
    """Normalization."""

    def __init__(self, mean=0., std=1.):
        self.mean = np.array(mean)
        self.std = np.array(std)
        if self.mean.ndim == 1:
            self.mean = self.mean[None, :, None, None]
        if self.std.ndim == 1:
            self.std = self.std[None, :, None, None]

    def __call__(self, x):
        return (x - self.mean) / self.std


# load pkl synthetic light-curve files to numpy array
class ProtoPlanetaryDisks(Dataset):
    """
    Dataset class that loads synthetic images of Protoplanetary disks,
    the dataset has shape [N, C, H, W] = [36518, 1, 187, 187]
    ...
    
    Attributes
    ----------
    imgs        : array
        array with images
    meta        : array
        array with physical parameters asociated to each image
    meta_names  : list
        list with the names of the physical parameters (8 columns)
    img_dim     : int
        image dimension, assuming square ratio
    img_channel : int
        number of channels per image
    transform   : bool
        apply rotation and flip transformation
    transform_fx : torchvision transformations
        set of transformations to be applyed when calling an item
    
    Methods
    -------
    __getitem__(self, index)
        return data in the index position, apply transform_fx if transform
        is True
    __len__(self)
        return the total length of the entire dataset
    get_dataloader(self, batch_size=32, shuffle=True,
                   test_split=0.2, random_seed=42)
        return a dataloader object for trainning and testing
    """
    def __init__(self, machine='local', transform=True,
                 subsample=False, img_norm=True):
        """
        Parameters
        ----------
        machine    : bool, optional
            which machine is been used (colab, exalearn, [local])
        transform  : bool, optional
            if apply or not image transformation when getting new item
        subsample  : bool, optional
            wheather to subsample the entire dataset, for fastloading and
            testing purposes
        img_norm   : bool, optional
            load images that are scaled to [0,1] when True, or raw images
            when False.
        """
        if machine == 'local':
            ppd_path = '%s/data/PPD' % (root)
        elif machine == 'colab':
            ppd_path = '%s/PPDAE' % (colab_root)
        elif machine == 'exalearn':
            ppd_path = '%s/PPD/' % (exalearn_root)
        else:
            raise('Wrong host, please select local, colab or exalearn')

        self.meta = np.load('%s/param_arr_gridandfiller1.npy' % (ppd_path))
        self.meta = self.meta.astype(np.float32)
        self.meta_names = ['m_dust', 'Rc', 'f_exp', 'H0',
                           'Rin', 'sd_exp', 'a_max', 'inc']
        self.img_norm = img_norm
        if not self.img_norm:
            self.imgs = np.load('%s/img_array.npy' % (ppd_path))
            self.imgs = np.expand_dims(self.imgs, axis=1)
        else:
            self.imgs = np.load('%s/img_array_gridandfiller1_norm.npy' % (ppd_path))
        self.imgs = self.imgs.astype(np.float32)

        if subsample:
            idx = np.where((self.meta[:,0] == 0.001) & 
                           (self.meta[:,1] == 300) &
                           (self.meta[:,2] == 1.3) &
                           (self.meta[:,3] == 20))[0]
            self.imgs = self.imgs[idx]
            self.meta = self.meta[idx]
        self.img_dim = self.imgs[0].shape[-1]
        self.img_channels = self.imgs[0].shape[0]
        self.transform = transform
        self.transform_fx = torchvision.transforms.Compose([
            MyRotationTransform(),
            MyFlipVerticalTransform()])


    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            positional index 
        Returns
        -------
            image and metadata at possition [index], applying set of 
            transform_fx if needed.
        """
        imgs = self.imgs[index]
        meta = self.meta[index]
        if self.transform:
            imgs = self.transform_fx(imgs)
        return imgs, meta

    def __len__(self):
        return len(self.imgs)

    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):
        """
        Parameters
        ----------
        batch_size : int
            size of each batch
        shuffle    : bool
            whether to shuffle or not the samples
        test_split : float
            fraction of the dataset to be used as test sample
        random_seed: int
            initialization of random seed
        
        Returns
        -------
        train_loader : 
            dataset loader with training instances 
        test_loader  : 
            dataset loader with testing instances 
        """
        np.random.seed(random_seed)
        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            test_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]
            del indices, split

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler,
                                      drop_last=False)
            test_loader = DataLoader(self, batch_size=batch_size,
                                     sampler=test_sampler,
                                     drop_last=False)

        return train_loader, test_loader


class MNIST(Dataset):
    """
    Dataset class that loads MNISt hand-writen digits,
    the dataset has shape [N, C, H, W] = [36518, 1, 187, 187]
    ...
    
    Attributes
    ----------
    train     : dataset
        train dataset
    test      : dataset
        test  dataset
    img_dim   : int
        image dimension, assuming square ratio
    
    Methods
    -------
    __getitem__(self, index)
        return data in the index position, apply transform_fx if transform
        is True
    __len__(self)
        return the total length of the entire dataset
    get_dataloader(self, batch_size=32, shuffle=True,
                   test_split=0.2, random_seed=42)
        return a dataloader object for trainning and testing
    """
    def __init__(self, machine='local'):
        if machine == 'local':
            mnist_path = '%s/data/' % (root)
        elif machine == 'colab':
            mnist_path = '%s/' % (colab_root)
        elif machine == 'exalearn':
            mnist_path = '%s/' % (exalearn_root)
        self.train = torchvision.datasets.MNIST(
            mnist_path, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.test = torchvision.datasets.MNIST(
            mnist_path, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.img_dim = self.test[0][0].shape[-1]
        self.img_channels = self.test[0][0].shape[0]

    def __getitem__(self, index):
        return self.test[index]

    def __len__(self):
        return len(self.train) + len(self.test)

    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=.2, random_seed=32):
        """
        Parameters
        ----------
        batch_size : int
            size of each batch
        shuffle    : bool
            whether to shuffle or not the samples
        test_split : float
            fraction of the dataset to be used as test sample
        random_seed: int
            initialization of random seed
        
        Returns
        -------
        train_loader : 
            dataset loader with training instances 
        test_loader  : 
            dataset loader with testing instances 
        """
        train_loader = DataLoader(self.train,
                                  batch_size=batch_size,
                                  shuffle=shuffle)

        test_loader = DataLoader(self.test,
                                 batch_size=batch_size,
                                 shuffle=shuffle)

        return train_loader, test_loader
