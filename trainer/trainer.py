import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, collate_fn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import sys
import h5py
import os 

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, cluster_loader,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        criterion = criterion(margin=config["trainer"]["margin"]**0.5)
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.cluster_loader = cluster_loader
        self.data_loader = data_loader
        self.loader_args = config['data_loader']['args']
        self.arch_args = config["arch"]["args"]
        self.train_args = config["trainer"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        # self.metric_ftns => metric functions list        
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        epoch_loss = 0
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # model을 train()으로 설정
        self.model.train()
        
        # train metric 초기화
        self.train_metrics.reset()
        
        # subsetN = 1
        # subsetIdx = [np.arange(self.len_epoch)]
        
        nBatches = (self.len_epoch + self.data_loader.batch_size -1) // self.data_loader.batch_size
        
        # for subIter in range(subsetN):
        print("===> Building Cache")
        self.model.eval()
        train_set_cache = os.path.join(self.train_args["cachePath"], self.config["mode"] + "_feat_cache.hdf5")
        with h5py.File(train_set_cache, mode='w') as h5:
            pool_size = self.arch_args["encoder_dim"]
            
            if self.config["arch"]["type"] == 'NetVLAD': pool_size *= self.arch_args["num_clusters"]
            
            # create h5 dataset to store features
            h5feat = h5.create_dataset("features",
                    [len(self.data_loader), pool_size],
                    dtype=np.float32)
            
            # push features into h5 dataset
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(self.cluster_loader, 1):
                    
                    input = input.to(self.device)
                    
                    image_encoding = self.model.encoder(input)
                    
                    vlad_encoding = self.model.pool(image_encoding)
                    
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    
                    del input, image_encoding, vlad_encoding
        
        # TODO train_set과 whole_train_set의 차이를 알아야할듯
        # => get_whole_training_set과 get_training_query_set의 차이를 알면 된다.
        # => WholeDatasetFromStruct vs QueryDatasetFromStruct
        # WholeDataset은 h5feat cluster를 생성하기 위해 사용
        # 리턴시 image만을 return, positive
        # train_set이 실제 train을 위한 데이터셋
        # sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        ### 06/26
        # training_data_loader = DataLoader(dataset=sub_train_set, num_workers=self.loader_args["num_workers"],
        #             batch_size=self.loader_args["batch_size"], shuffle=self.loader_args["shuffle"],
        #             collate_fn=collate_fn, pin_memory=self.loader_args["cuda"])
        
        training_data_loader = self.data_loader
        
        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())
        
        self.model.train()
        
        # data => 데이터, target => GT
        for iteration, (query, positives, negatives,
                negCounts, indices) in enumerate(training_data_loader, startIter):
            
            if query is None: continue
            
            B, C, H, W = query.shape
            
            # negCounts는 각 query에 대한 negative의 개수를 나타내는 list
            # nNeg는 negative의 총 개수
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])
            
            input = input.to(self.device)
            image_encoding = self.model.encoder(input)
            vlad_encoding = self.model.pool(image_encoding)

            # split vlad encoding into query, positive, negative                
            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])
            
            self.optimizer.zero_grad()
            
            # loss 계산 & 파라미터 조정
            # 각각의 query, positive, negative triplet에 대해 loss를 계산
            # TODO
            # negative의 개수가 달라질 수 있기 때문에(?) query, negative 마다 loss를 계산해야 한다.
            loss = 0
            
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i])+n).item()
                    loss += self.criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])
            
            # normalize by actual number of negatives
            loss /= nNeg.float().to(self.device)
            loss.backward()
            self.optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # TODO
            # # steps_per_sec 기록
            # self.writer.set_step((epoch - 1) * self.len_epoch + iteration)
            
            # # writer에 'loss'라는 이름으로 loss 값 기록
            # self.train_metrics.update('batch_loss', loss.item())
            
            # # metric_ftns에 담긴 met 각각에 대해 값 계산 후 기록
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            # # logger로 loss 출력 및 image 기록
            # if batch_idx % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.item()))
            #     # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # if batch_idx == self.len_epoch:
            #     break

        startIter += len(self.data_loader)
        del self.data_loader, loss
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        os.remove(train_set_cache) # delete HDF5 cache
    
        avg_loss = epoch_loss / nBatches
        
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
                flush=True)
                    
        # # {'loss': 0.658925260342128, 'accuracy': 0.7893587085308057, 'top_k_acc': 0.9289205314827352}            
        # log = self.train_metrics.result()

        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     # output metric 출력
        #     log.update(**{'val_'+k : v for k, v in val_log.items()})

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        # return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # model 내 layer별 파라미터 기록
        # name => layer 이름, p => parameter
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        
        # TODO
        # Mini-batch 여부를 말하는 듯?
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
