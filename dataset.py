from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.path = path
        self.resolution = resolution
        self.transform = transform
        self.blacklist = np.array([40650])
        env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            raise IOError('Cannot open lmdb dataset', self.path)
        with env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            print(f"begin load  data {self.length}")
        env.close()
    
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
            self.txn = txn




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
        # index = self.get_index(idx)
        index = idx
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


if __name__ == '__main__':
    dataset = MultiResolutionDataset("sexyface",transforms.Compose([]),resolution=256)
    dataset.open_lmdb()
    dataset.get_index(0)
    dataset.check_consistency()