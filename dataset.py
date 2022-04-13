from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

import numpy as np


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.path = path
        self.resolution = resolution
        self.transform = transform
        self.blacklist = np.array([40650])
        self.length = 136723
        # self.length = 1000
        # self.check_consistency()
    
    def open_lmdb(self):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', self.path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.txn = txn
        self.length -= len(self.blacklist)
        # print(f'MultiResolutionDataset len: {self.length}')



    def get_index(self, idx):
        shift = sum(self.blacklist <= idx)
        return idx + shift
    
    def check_consistency(self):
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            try:
                img = Image.open(buffer)
            except:
                print(f'Exception at {index}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        index = self.get_index(idx)
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img