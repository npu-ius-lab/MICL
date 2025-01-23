import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from tqdm import tqdm 
from eval.evaluate import evaluate 
from losses.loss_factory import make_inc_loss,get_cp_loss,get_new_loss,get_seen_loss
from misc.util import AverageMeter
from models.model_factory import model_factory
from datasets.dataset_util_inc import make_dataloader
from torchpack.utils.config import configs


class TrainerIncremental(nn.Module):
    def __init__(self, logger, memory, old_environment_pickle, new_environment_pickle, pretrained_checkpoint, env_idx):
        # Initialise inputs
        super(TrainerIncremental, self).__init__()
        self.debug = configs.debug 
        self.logger = logger 
        self.env_idx = env_idx
        self.epochs = configs.train.optimizer.epochs 

        # Set up meters and stat trackers 
        self.loss_total_meter = AverageMeter()
        self.loss_contrastive_meter = AverageMeter()
        self.loss_inc_meter = AverageMeter()
        self.positive_score_meter = AverageMeter()
        self.negative_score_meter = AverageMeter()
        self.positive_score_meter_mem = AverageMeter()
        self.negative_score_meter_mem = AverageMeter()
        # contrastive
        self.K = 1000
        self.m = 0.99
        self.T = 0.07
        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = new_environment_pickle, memory = memory)

        # Build models and init from pretrained_checkpoint
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model_frozen = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new_q = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new_k = model_factory(ckpt = None, device = 'cuda')
        

        if configs.train.loss.incremental.name == 'MI':
            self.cp_loss = get_cp_loss()
            self.new_loss = get_new_loss()
            self.seen_loss = get_seen_loss()
            
            self.cp_loss_meter = AverageMeter()
            self.new_loss_meter = AverageMeter()
            self.seen_loss_meter = AverageMeter()

            if configs.train.uncertainty_weight:
                self.loss_weights = nn.Parameter(torch.ones(4))
                self.weight_0 = AverageMeter()
                self.weight_1 = AverageMeter()
                self.weight_2 = AverageMeter()
                self.weight_3 = AverageMeter()  
              

        for param_q, param_k in zip(self.model_new_q.parameters(), self.model_new_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue_pcd', torch.randn(128, len(self.dataloader.dataset.queries)))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0).cuda()
        self.queue_pcd_index = set(range(len(self.dataloader.dataset.queries)))
        print("loading all embeddings")
        # loading queue
        for idx, (queries, keys, memories, labels,_,_,_,_) in enumerate(self.dataloader):
            query_size = int((len(labels) - configs.train.sample_pair_num * 2))
            # print(query_size)
            queries = {x: queries[x].to('cuda') if x!= 'coords' else queries[x] for x in queries}
            with torch.no_grad():  # no gradient to keys
                embeddings, projectors = self.model_new_k(queries)
                projectors = nn.functional.normalize(projectors, dim=1)
            self._dequeue_and_enqueue_pcd_fast(projectors, labels[:query_size])
        print(f"finish constructing incremental_environments,size: {len(self.dataloader.dataset.queries)}")
        # Make optimizer 
        if configs.train.optimizer.name == "SGD":
            if configs.train.uncertainty_weight:
                self.optimizer = torch.optim.SGD([
                                                {"params": self.model_new_q.parameters()},
                                                {"params": [self.loss_weights]}  
                                                ], lr=configs.train.optimizer.lr,momentum=configs.train.optimizer.momentum,
                                                weight_decay=configs.train.optimizer.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model_new_q.parameters(), lr=configs.train.optimizer.lr,
                                            momentum=configs.train.optimizer.momentum,
                                            weight_decay=configs.train.optimizer.weight_decay)
        elif configs.train.optimizer.name == "Adam":
            if configs.train.uncertainty_weight:
                self.optimizer = torch.optim.Adam([{"params": self.model_new_q.parameters()},{"params": [self.loss_weights]}], lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model_new_q.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

        # Scheduler
        if configs.train.optimizer.scheduler is None:
            self.scheduler = None
        else:
            if configs.train.optimizer.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs.train.optimizer.epochs+1,
                                                                    eta_min=configs.train.optimizer.min_lr)
            elif configs.train.optimizer.scheduler == 'MultiStepLR':
                if not isinstance(configs.train.optimizer.scheduler_milestones, list):
                    configs.train.optimizer.scheduler_milestones = [configs.train.optimizer.scheduler_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, configs.train.optimizer.scheduler_milestones, gamma=0.1)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(configs.train.optimizer.scheduler))

        # Make loss functions
        self.inc_fn = make_inc_loss()
        
        self.criterion = nn.CrossEntropyLoss().cuda()


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_new_q.parameters(), self.model_new_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pcd_fast(self, keys, labels):
        self.queue_pcd[:, labels] = keys.T

    '''
        reset epoch metrics
    '''
    def before_epoch(self, epoch):
        # Reset meters
        self.loss_total_meter.reset()
        self.loss_inc_meter.reset()
        self.loss_contrastive_meter.reset()
        self.positive_score_meter.reset()
        self.negative_score_meter.reset()

        self.positive_score_meter_mem.reset()
        self.negative_score_meter_mem.reset()

        # Adjust weight of incremental loss function if constraint relaxation enabled
        self.inc_fn.adjust_weight(epoch)
        if configs.train.uncertainty_weight and configs.train.loss.incremental.name == 'MI':
            self.seen_loss_meter.reset()
            self.new_loss_meter.reset()
            self.cp_loss_meter.reset()
            if configs.train.uncertainty_weight:
                self.weight_0.reset()
                self.weight_1.reset()
                self.weight_2.reset()
                self.weight_3.reset()
    '''
        1. clear the gradient of weights -------> self.optimizer.zero_grad()
        2. model forward
        3. calculate the loss
        4. calculate the gradients -------------> loss.backward()
        5. update the weights ------------------> self.optimizer.step()
        6. empty CUDA cache
        7. update epoch metrics
    # '''
    def training_step(self, queries, keys, memories, labels,mem_pos_mask,mem_neg_mask,new_pos_mask,new_neg_mask):
        memories = {x: memories[x].to('cuda') if x!= 'coords' else memories[x] for x in memories}
        queries = {x: queries[x].to('cuda') if x!= 'coords' else queries[x] for x in queries}
        keys = {x: keys[x].to('cuda') if x!= 'coords' else keys[x] for x in keys}
        anchor_size = int((len(labels) - configs.train.sample_pair_num * 2) / 2)
        
        key_labels = labels[anchor_size:2*anchor_size] + labels[:anchor_size]
        
        # Get embeddings and Loss
        self.optimizer.zero_grad()
        embeddings, projectors = self.model_new_q(queries)

        projectors = nn.functional.normalize(projectors, dim=1)
        embeddings_memories, projectors_memories = self.model_new_q(memories)
        projectors_memories = nn.functional.normalize(projectors_memories, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, key_projectors_new = self.model_new_k(keys)
            key_projectors_new = nn.functional.normalize(key_projectors_new, dim=1)
            
            _, key_mem_projectors_new = self.model_new_k(memories)
            key_mem_projectors_new = nn.functional.normalize(key_mem_projectors_new, dim=1)
            
            embeddings_memories_frozen, projectors_memories_frozen = self.model_frozen(memories)
            projectors_memories_frozen = nn.functional.normalize(projectors_memories_frozen, dim=1)
        
        #add mem contrastive
        queue_pcd_clone = self.queue_pcd.clone().detach()
        anchor_size_mem = configs.train.sample_pair_num 
        queries_projectors_new_mem = torch.cat((projectors_memories[:anchor_size_mem],projectors_memories[anchor_size_mem:]))
        key_projectors_new_mem = torch.cat((key_mem_projectors_new[anchor_size_mem:],key_mem_projectors_new[:anchor_size_mem]))
        l_pos_pc_mem = torch.einsum('nc,nc->n', [queries_projectors_new_mem, key_projectors_new_mem]).unsqueeze(-1)
        

        negatives_list = []
        
        ## mem contrastive
        memory_indexes = random.sample(list(self.dataloader.dataset.queries), len(labels[2*anchor_size:]))
        for memory_index in memory_indexes:
            negative_index = random.sample(list(self.queue_pcd_index.difference(set(self.dataloader.dataset.queries[memory_index].non_negatives))), self.K)
            negatives_list.append(queue_pcd_clone[:,negative_index])
        
        negatives_tensor = torch.stack(negatives_list, dim=0)
        l_neg_pcd_mem = torch.einsum('nc,nck->nk', [queries_projectors_new_mem, negatives_tensor])
        
        positive_score_mem = torch.mean(l_pos_pc_mem)
        negative_score_mem = torch.mean(l_neg_pcd_mem)
        
        logits_pcd_mem = torch.cat([l_pos_pc_mem, l_neg_pcd_mem], dim=1)
        logits_pcd_contrastive_mem = logits_pcd_mem / self.T
        labels_pcd_mem = torch.zeros(logits_pcd_contrastive_mem.shape[0], dtype=torch.long).cuda()
        loss_contrastive_mem = self.criterion(logits_pcd_contrastive_mem, labels_pcd_mem)

        

        # compute logits
        # positive logits: Nx1
        l_pos_pcd = torch.einsum('nc,nc->n', [projectors, key_projectors_new]).unsqueeze(-1)

        negatives_list = []
        for index in range(len(labels[:2*anchor_size])):
            negative_index = random.sample(list(self.queue_pcd_index.difference(set(self.dataloader.dataset.queries[labels[index]].non_negatives))), self.K)
            negatives_list.append(queue_pcd_clone[:,negative_index])
        negatives_tensor = torch.stack(negatives_list, dim=0)
        # negative logits: NxK
        l_neg_pcd = torch.einsum('nc,nck->nk', [projectors, negatives_tensor])
        positive_score = torch.mean(l_pos_pcd)
        negative_score = torch.mean(l_neg_pcd)
        
        logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd], dim=1)
        logits_pcd_contrastive = logits_pcd / self.T
        labels_pcd = torch.zeros(logits_pcd_contrastive.shape[0], dtype=torch.long).cuda()
        loss_contrastive_new = self.criterion(logits_pcd_contrastive, labels_pcd)

        loss_contrastive = loss_contrastive_new + loss_contrastive_mem # add

        if configs.train.loss.incremental.name == 'KD':
            loss_incremental = self.inc_fn(projectors_memories_frozen, projectors_memories)
        elif configs.train.loss.incremental.name == 'MI':
            loss_cp = self.cp_loss(projectors_memories_frozen, projectors_memories, mem_pos_mask,mem_neg_mask)
            loss_seen = self.seen_loss(projectors_memories,mem_pos_mask,mem_neg_mask)
            loss_new = self.new_loss(projectors,new_pos_mask,new_neg_mask)
            loss_incremental = loss_cp + loss_seen + loss_new
            if  not configs.train.uncertainty_weight:
                loss_incremental = loss_cp + loss_seen + loss_new
        else:
            print('error')


        if not configs.train.uncertainty_weight:
            loss_total = loss_contrastive + loss_incremental 
        else:
            loss_list = [loss_contrastive, loss_cp, loss_seen, loss_new]
            final_loss = []
            for i in range(len(self.loss_weights)):
                final_loss.append(loss_list[i] / (2 * self.loss_weights[i].pow(2)) + torch.log(self.loss_weights[i]))
            loss_total = torch.sum(torch.stack(final_loss))
        # dequeue and enqueue
        self._dequeue_and_enqueue_pcd_fast(key_projectors_new, key_labels)
        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_total_meter.update(loss_total.item())
        self.loss_contrastive_meter.update(loss_contrastive.item())

        if configs.train.uncertainty_weight and configs.train.loss.incremental.name == 'MI':
            self.seen_loss_meter.update(loss_seen.item())
            self.new_loss_meter.update(loss_new.item())
            self.cp_loss_meter.update(loss_cp.item())
            if configs.train.uncertainty_weight:
                self.weight_0.update(self.loss_weights[0].item())
                self.weight_1.update(self.loss_weights[1].item())
                self.weight_2.update(self.loss_weights[2].item())
                self.weight_3.update(self.loss_weights[3].item())
        
        self.loss_inc_meter.update(loss_incremental.item())
        self.positive_score_meter.update(positive_score.item())
        self.negative_score_meter.update(negative_score.item())

        self.positive_score_meter_mem.update(positive_score_mem.item())
        self.negative_score_meter_mem.update(negative_score_mem.item())
    '''
        # 1. update learning rate
        # 2. update batch size
        # 3. save log data
    '''
    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()

        # Tensorboard plotting
        self.logger.add_scalar(f'Step_{self.env_idx}/Total_Loss_epoch', self.loss_total_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Contrastive_Loss_epoch', self.loss_contrastive_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Increment_Loss_epoch', self.loss_inc_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/positive_score_epoch', self.positive_score_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/negative_score_epoch', self.negative_score_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/positive_score_mem_epoch', self.positive_score_meter_mem.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/negative_score_mem_epoch', self.negative_score_meter_mem.avg, epoch)
        
        if configs.train.uncertainty_weight and configs.train.loss.incremental.name == 'MI':
            self.logger.add_scalar(f'Step_{self.env_idx}/Seen_loss', self.seen_loss_meter.avg, epoch)
            self.logger.add_scalar(f'Step_{self.env_idx}/New_loss', self.new_loss_meter.avg, epoch)
            self.logger.add_scalar(f'Step_{self.env_idx}/Cp_loss', self.cp_loss_meter.avg, epoch)
            if configs.train.uncertainty_weight:
                self.logger.add_scalar(f'Step_{self.env_idx}/Place_Rec_Loss_weight', self.weight_0.avg, epoch)
                self.logger.add_scalar(f'Step_{self.env_idx}/Cp_loss_weight', self.weight_1.avg, epoch)
                self.logger.add_scalar(f'Step_{self.env_idx}/Seen_loss_weight', self.weight_2.avg, epoch)
                self.logger.add_scalar(f'Step_{self.env_idx}/New_loss_weight', self.weight_3.avg, epoch)


    # for loop on epochs
    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (queries, keys, memories, labels,positives_mask,negatives_mask,positives_mask_new,negatives_mask_new) in enumerate(self.dataloader):
                self.training_step(queries, keys, memories, labels,positives_mask,negatives_mask,positives_mask_new,negatives_mask_new)
            self.after_epoch(epoch)
        return self.model_new_q
