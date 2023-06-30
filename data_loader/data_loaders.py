from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import *
import torch.utils.data as data

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

"""
1. dataloader를 indexing시 얻어야 하는 것 : (query, positives, negatives, negCounts, indices)
2. collate_fn 활용 : 단일이미지를 리턴하는게 아니라 tuple를 리턴한다!

데이터를 받아서 전달하는 역할만 수행
DataLoader()에서 전달하는 args만 받아서 전달하면 될듯

"""

class ClusterLoader(BaseDataLoader):
    def __init__(self,structFile,batch_size, shuffle,num_workers,validation_split):

        self.dataset = get_whole_training_set(structFile,False)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        super().__init__(dataset=self.dataset,batch_size=self.batch_size,shuffle=self.shuffle,
                         collate_fn=data.dataloader.default_collate,num_workers=self.num_workers,validation_split=validation_split)

class TrainLoader(BaseDataLoader):
    def __init__(self,structFile,margin,nNegSample,nNeg,batch_size, shuffle,num_workers,validation_split):

        self.dataset = get_training_query_set(structFile,margin,nNegSample,nNeg)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        super().__init__(dataset=self.dataset,batch_size=self.batch_size,shuffle=self.shuffle,
                         collate_fn=data.dataloader.default_collate,num_workers=self.num_workers,validation_split=validation_split)
    