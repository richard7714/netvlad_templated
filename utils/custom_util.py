import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler
from os.path import join, exists
from os import makedirs
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

import pandas as pd

from sklearn.neighbors import NearestNeighbors
import h5py

import faiss
import os
import sys

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

def torch_cat(negs,dim):
    return torch.cat(negs,dim)

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640,480)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
def get_whole_training_set(structFile,onlyDB=False):
    return WholeDatasetFromStruct(structFile,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)
    
def get_whole_val_set(structFile):
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(structFile,*train_args):
    return QueryDatasetFromStruct(structFile,train_args,
                             input_transform=input_transform())

def get_val_query_set(structFile,root_dir):
    return QueryDatasetFromStruct(structFile,root_dir,
                             input_transform=input_transform())


def parse_dbStruct(path):
    database = pd.read_csv(path)

    dataset = "sthereo"

    whichSet = "train"

    dbImage = [row for row in database["db_image"]]
        
    utmDb = np.array([database["db_pose_x"].astype(float).tolist(),database["db_pose_y"].astype(float).tolist()])
    utmDb = utmDb.T
    
    numDb = database["dbNum"][0].astype(int)
    numQ = database["qNum"][0].astype(int)
    
    qImage = [row for row in database["q_image"] if isinstance(row, str)]
    utmQ = np.array([database["q_pose_x"][:numQ].astype(float).tolist(),database["q_pose_y"][:numQ].astype(float).tolist()])
    utmQ = utmQ.T
    
    posDistThr = database["thres"][0].astype(float)
    posDistSqThr = database["sqthres"][0].astype(float)
    nonTrivPosDistSqThr = 10

    dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)    

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile,input_transform=None, onlyDB=False):
        super().__init__()
        
        self.input_transform = input_transform
        
        self.dbStruct = parse_dbStruct(structFile)
        self.images = [dbIm for dbIm in self.dbStruct.dbImage if isinstance(dbIm, str)]
        if not onlyDB:
            self.images += [qIm for qIm in self.dbStruct.qImage if isinstance(qIm, str)]
                
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None
    
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
                       
        if self.input_transform:
            img = self.input_transform(img)
        
        return img, index
        
    def __len__(self):
        return len(self.images)
    
    def getPositives(self):
        # positives for evaluation are those within threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            
            # utmDb에 대해 knn을 학습시키고 utmQ에 대해 knn을 적용하여 self.dbStruct.posDistThr 이내의 데이터를 얻는다.
            knn.fit(self.dbStruct.utmDb)
            
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)
        
        return self.positives

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile,*train_args, input_transform=None):
        super().__init__()
        
        self.input_transform = input_transform
        self.margin = train_args[0][0]
        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = train_args[0][1] # number of negatives to randomly sample
        self.nNeg = train_args[0][2] # number of negatives used for training
        
        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(self.dbStruct.utmDb)
                
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5,
                return_distance=False))
        
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        
        # it's possible some queries don't have any non-trivial potential positives
        # let's filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]
        
        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr,
                return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))
        
        self.cache = "./cache/train_feat_cache.hdf5" # filepath of HDF5 containing feature vectors for images
        
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]
        
    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")
            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index+qOffset]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(posFeat)
            
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()
            
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
            
            negFeat = h5feat[negSample.astype(int).tolist()]
            knn.fit(negFeat)
            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1),
                    self.nNeg*10)
            
            # 왜함? TODO
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)
            
            # try to find negtavies that are within margin, if there aren't any, return None
            violatingNeg = dNeg < dPos + self.margin ** 0.5
            
            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices
        
        query = Image.open(self.dbStruct.qImage[index])
        positive = Image.open(self.dbStruct.dbImage[posIndex])
            
        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)
            
        negatives = []
        for negIndex in negIndices:
            negative = Image.open(self.dbStruct.dbImage[negIndex])
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)
        
        negatives = torch.stack(negatives,0)
        
        return query, positive, negatives, [index, posIndex]+negIndices.tolist()
    
    def __len__(self):
        return len(self.queries)
    
def get_clusters(model,cluster_set,config):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)
    cluster_args = config["cluster_loader"]["args"]
    device = "cuda" if config["n_gpu"] > 0 else "cpu"
    encoder_dim = config["pool"]["args"]["dim"]
    num_clusters = config["pool"]["args"]["num_clusters"]
        
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=cluster_args["num_workers"], batch_size=cluster_args["cacheBatchSize"], shuffle=cluster_args["shuffle"], 
                pin_memory=cluster_args["pin_memory"],
                sampler=sampler)

    if not exists('centroids'):
        makedirs('centroids')
    
    initcache = join('centroids', cluster_set.dataset + '_' + str(num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                
                input = input.to(device)

                input = input.float()
                                                    
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*cluster_args["cacheBatchSize"]*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/cluster_args["cacheBatchSize"])), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
