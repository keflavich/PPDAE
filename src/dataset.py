import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from sklearn import preprocessing

root = '/Users/jorgetil/Astro/PPD-AE'
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
    def __init__(self, machine='exalearn', transform=True,
                 query='', par_norm=False):
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
            ppd_path = '%s/data/PPD/partitions' % (root)
        elif machine == 'colab':
            ppd_path = '%s/PPDAE/partitions' % (colab_root)
        elif machine == 'exalearn':
            ppd_path = '%s/PPD/partitions' % (exalearn_root)
        else:
            raise('Wrong host, please select local, colab or exalearn')

        self.par_train = np.load('%s/param_arr_gridandfiller123%s_train_all.npy' % 
                            (ppd_path, '_norm' if par_norm else ''))
        self.par_test = np.load('%s/param_arr_gridandfiller123%s_test.npy' % 
                            (ppd_path, '_norm' if par_norm else ''))
        self.par_names = ['m_dust', 'Rc', 'f_exp', 'H0',
                           'Rin', 'sd_exp', 'a_max', 'inc']

        self.imgs_train = np.load('%s/img_array_gridandfiller123_norm_train_all.npy' % 
                                  (ppd_path))
        self.imgs_test = np.load('%s/img_array_gridandfiller123_norm_test.npy' % 
                                  (ppd_path))

        if query != '':
            aux_df = pd.DataFrame(self.par_train, columns=self.par_names)
            idx = aux_df.query(query).index
            del aux_df
            self.imgs_train = self.imgs_train[idx]
            self.par_train = self.par_train[idx]
            
        self.img_dim = self.imgs_test[0].shape[-1]
        self.img_channels = self.imgs_test[0].shape[0]
        self.transform = transform
        self.transform_fx = torchvision.transforms.Compose([
            MyRotationTransform(),
            MyFlipVerticalTransform()])
        self.par_norm = par_norm
        self.MinMaxSc = preprocessing.MinMaxScaler()
        self.MinMaxSc.fit(np.concatenate([self.par_train, self.par_test]))


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
        img = self.imgs_train[index]
        par = self.par_train[index]
        if self.transform:
            img = self.transform_fx(img)
        if self.par_norm:
            par = self.MinMaxSc.transform(par.reshape(1, -1))[0]
        return img, par

    
    def __len__(self):
        return return len(self.par_train) + len(self.par_test)

    
    def get_dataloader(self, batch_size=32, shuffle=True,
                       val_split=0.2, random_seed=42):
        """
        Parameters
        ----------
        batch_size : int
            size of each batch
        shuffle    : bool
            whether to shuffle or not the samples
        val_split : float
            fraction of the dataset to be used as validation sample
        random_seed: int
            initialization of random seed
        
        Returns
        -------
        train_loader : 
            dataset loader with training instances 
        val_loader  : 
            dataset loader with validation instances 
        test_loader  : 
            dataset loader with testing instances 
        """
        np.random.seed(random_seed)
        if val_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            val_loader = None
        else:
            # Creating data indices for training and val splits:
            dataset_size = len(self.par_train)
            indices = list(range(dataset_size))
            split = int(np.floor(val_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            del indices, split

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler,
                                      drop_last=False)
            val_loader = DataLoader(self, batch_size=batch_size,
                                     sampler=val_sampler,
                                     drop_last=False)
            
        if self.par_norm:
            aux_par_test = self.MinMaxSc.transform(self.par_test)
        else:
            aux_par_test = self.par_test
        test_ds = TensorDataset(torch.Tensor(self.imgs_test),
                                torch.Tensor(aux_par_test))
        del aux_par_test
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 drop_last=False)

        return train_loader, val_loader, test_loader


    
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
