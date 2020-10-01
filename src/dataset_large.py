import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from bisect import bisect
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
                 img_norm=True):
        """
        Parameters
        ----------
        machine    : bool, optional
            which machine is been used (colab, exalearn, [local])
        transform  : bool, optional
            if apply or not image transformation when getting new item
        img_norm   : bool, optional
            load images that are scaled to [0,1] when True, or raw images
            when False.
        """
        if machine == 'local':
            ppd_path = '%s/data/PPD' % (root)
        elif machine == 'colab':
            ppd_path = '%s/PPDAE/partitions' % (colab_root)
        elif machine == 'exalearn':
            ppd_path = '%s/PPD/partitions' % (exalearn_root)
        else:
            raise('Wrong host, please select local, colab or exalearn')
            
        self.par_paths = sorted(glob.glob('%s/param_arr_gridandfiller123_train_*.npy' %
                                  (ppd_path)))
        self.par_memmaps = [np.load(path, mmap_mode='r') for path in self.par_paths]
        
        self.imgs_paths = sorted(glob.glob('%s/img_array_gridandfiller123_norm_train_*.npy' %
                                   (ppd_path)))
        self.imgs_memmaps = [np.load(path, mmap_mode='r') for path in self.imgs_paths]
        self.start_indices = [0] * len(self.imgs_paths)
        self.data_count = 0
        for index, memmap in enumerate(self.imgs_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        
        
        self.par_names = ['m_dust', 'Rc', 'f_exp', 'H0',
                           'Rin', 'sd_exp', 'a_max', 'inc']
        self.par_test = np.load('%s/param_arr_gridandfiller123_test.npy' % (ppd_path))
        self.imgs_test = np.load('%s/img_array_gridandfiller123_norm_test.npy' % (ppd_path))

        self.img_dim = self.imgs_test[0].shape[-1]
        self.img_channels = self.imgs_test[0].shape[0]
        self.transform = transform
        self.transform_fx = torchvision.transforms.Compose([
            MyRotationTransform(),
            MyFlipVerticalTransform()])
        

    def __len__(self):
        return self.data_count
    

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        img = self.imgs_memmaps[memmap_index][index_in_memmap]
        par = self.par_memmaps[memmap_index][index_in_memmap]
        if self.transform:
            img = self.transform_fx(img)
        return np.array(img), np.array(par)
    

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
        """
        np.random.seed(random_seed)
        if val_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            val_loader = None
        else:
            # Creating data indices for training and val splits:
            dataset_size = len(self)
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

        return train_loader, val_loader