import numpy as np
import torch
from torchvision.utils import make_grid
from math import ceil
from base import BaseTrainer
from utils import inf_loop, MetricTracker, collate_fn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import sys
import h5py
import os 
from tqdm import tqdm
import faiss
from PIL import Image
from torchvision.transforms import ToTensor

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, cluster_loader,
                 train_loader, valid_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.cluster_loader = cluster_loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.train_set = train_loader.get_dataset()
        
        self.loader_args = config['train_loader']['args']
        self.pool_args = config["pool"]["args"]
        self.train_args = config["trainer"]
        self.valid_args = config["valid_loader"]["args"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch
        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_loader.batch_size))
        self.margin = self.loader_args["margin"]
        
        # self.metric_ftns => metric functions list        
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        initcache = os.path.join('centroids', config["dataset"]+'_'+str(self.pool_args["num_clusters"])+'_desc_cen.hdf5')
        
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            self.model.pool.init_params(clsts, traindescs)
            self.model.pool.to(device)
            del clsts, traindescs

    def _train_epoch(self, epoch):
        epoch_loss = 0
        startIter = 1
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
        
        if self.train_args["cacheRefreshRate"] > 0:
            # split train set into subsets
            subsetN = ceil(len(self.train_loader)/ self.train_args["cacheRefreshRate"])
            
            # split indices into subsets
            subsetIdx = np.array_split(np.arange(len(self.train_loader)),subsetN)
        else:
            subsetN = 1
            subsetIdx = [np.arange(len(self.train_loader))]
        
        # iterate over subsets
        nBatches = (len(self.train_loader) + self.loader_args["batch_size"]-1) // self.loader_args["batch_size"]
                
        for subIter in range(subsetN):
            print("===> Building Cache")
            self.model.eval()
            train_set_cache = os.path.join(self.train_args["cachePath"], self.config["mode"] + "_feat_cache.hdf5")
            with h5py.File(train_set_cache, mode='w') as h5:
                pool_size = self.pool_args["dim"]
                
                if self.config["pool"]["type"] == 'NetVLAD': pool_size *= self.pool_args["num_clusters"]
                
                # create h5 dataset to store features
                h5feat = h5.create_dataset("features",
                        [len(self.cluster_loader), pool_size],
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
                        
            sub_train_set = Subset(dataset=self.train_set, indices=subsetIdx[subIter])

            ### 06/26
            training_data_loader = DataLoader(dataset=sub_train_set, num_workers=self.loader_args["num_workers"],
                        batch_size=self.loader_args["batch_size"], shuffle=self.loader_args["shuffle"],
                        collate_fn=collate_fn, pin_memory=self.loader_args["pin_memory"])
            
            # training_data_loader = self.train_loader
            
            print('Allocated:', torch.cuda.memory_allocated())
            print('Cached:', torch.cuda.memory_cached())
            
            self.model.train()
            
            # data => 데이터, target => GT
            # for iteration, (query, positives, negatives,
            #         negCounts, indices) in enumerate(training_data_loader, startIter):

            for iteration, (query, positives, negatives,
                    negCounts,indices) in enumerate(training_data_loader, startIter):

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
                        loss += self.criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1],self.margin)
                
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
                if iteration % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(iteration),
                        loss.item()))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # if batch_idx == self.len_epoch:
                #     break

            startIter += len(training_data_loader)
            del training_data_loader, loss
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            os.remove(train_set_cache) # delete HDF5 cache
    
        avg_loss = epoch_loss / nBatches
        
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
                flush=True)
                    
        # # {'loss': 0.658925260342128, 'accuracy': 0.7893587085308057, 'top_k_acc': 0.9289205314827352}            
        log = self.train_metrics.result()
        log = {"loss" : avg_loss}
        # log = avg_loss
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            # output metric 출력
            log.update(**{'Recall@'+str(k) : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        eval_set = self.valid_loader.get_dataset()
        valid_loader = DataLoader(dataset=eval_set,
                                  num_workers = self.valid_args["num_workers"], batch_size=self.valid_args["cacheBatchSize"],
                                  shuffle=self.valid_args["shuffle"],pin_memory=self.valid_args["pin_memory"])
        # self.valid_metrics.reset()
        with torch.no_grad():
            print('====> Extracting Features')
            pool_size = self.pool_args["dim"]
            if self.config["pool"]["type"].lower() == "netvlad" : pool_size *= self.pool_args["num_clusters"]
            dbFeat = np.empty((len(eval_set),pool_size))
            
            for iteration, (input, indices) in enumerate(valid_loader, 1):
                
                input = input.to(self.device)

                image_encoding = self.model.encoder(input)
                vlad_encoding = self.model.pool(image_encoding)
                
                dbFeat[indices.detach().numpy(),:] = vlad_encoding.detach().cpu().numpy()
                
                del input, image_encoding, vlad_encoding
            del valid_loader
            
            # extracted for both db and query, now split in own sets
            qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
            dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
                
            print('====> Building faiss index')
            faiss_index = faiss.IndexFlatL2(pool_size)
            faiss_index.add(dbFeat)
            
            print('====> Calculating recall @ N')
            n_values = [1, 5, 10, 20]
            
            _, predictions = faiss_index.search(qFeat, max(n_values))
            
            # for each query get those within threshold distance
            gt = eval_set.getPositives()
            
            correct_at_n = np.zeros(len(n_values))
            
            for qIx, pred in enumerate(predictions):
                for i,n in enumerate(n_values):
                    compare = np.in1d(pred[:n], gt[qIx])
                    if np.any(compare):
                        correct_at_n[i:] += 1
                        break
                    else:
                        idx = np.where(compare == False)[0][0]
                        # TODO to be replaced into flag
                        # if True:
                        #     query_img = Image.open(eval_set.dbStruct.qImage[qIx])
                        #     self.writer.add_image(str(epoch)+'_query_img',ToTensor()(query_img), qIx)
                        #     q_img = Image.open(eval_set.dbStruct.dbImage[predictions[qIx,idx]])
                        #     self.writer.add_image(str(epoch)+'_answer',ToTensor()(q_img),qIx)
                        break
            
            recall_at_n = correct_at_n / eval_set.dbStruct.numQ
            
            recalls = {}
            for i,n in enumerate(n_values):
                recalls[n] = recall_at_n[i]
                # print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
                # if True:
                #     self.writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)
            
            return recalls
            # self.writer.set_step((epoch - 1) * len(self.valid_loader) + batch_idx, 'valid')
            # self.valid_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.valid_metrics.update(met.__name__, met(output, target))
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

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
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = len(self.train_loader)
        else:
            current = batch_idx
            # total = self.len_epoch
            total = (self.len_epoch + self.loader_args["batch_size"]-1) // self.loader_args["batch_size"]

        return base.format(current, total, 100.0 * current / total)
