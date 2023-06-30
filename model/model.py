import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(BaseModel):
    """NetVLAD layer implementation"""
    
    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False):
        
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larget value is harder assignment
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        # BaseModel의 init을 실행시키기 위함
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    
    def init_params(self, clsts, traindescs):
        
        # TODO
        if self.vladv2 == False:
            clstAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1,:]
            
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            # cluster 만드는 부분인듯?
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts,2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1]-dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq
            
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )
    
    def forward(self, x):
        N, C = x.shape[:2]
        
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) # across descriptor dim
        
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)
        
        # intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2) 
        
        # flatten
        vlad = vlad.view(x.size(0), -1) 
        
        # L2 normalize
        vlad = F.normalize(vlad, p=2, dim=1) 
        
        return vlad
        