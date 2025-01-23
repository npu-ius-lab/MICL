import torch
import torch.nn.functional as F 

import numpy as np 
from tqdm import tqdm 

from misc.utils import AverageMeter
from models.model_factory import model_factory
from datasets.dataset_utils_inc import make_dataloader
from losses.loss_factory import make_pr_loss, make_inc_loss,get_cp_loss,get_new_loss,get_seen_loss
from torchpack.utils.config import configs 
import torch.nn as nn



class TrainerIncremental:
    def __init__(self, logger, memory, old_environment_pickle, new_environment_pickle, pretrained_checkpoint, env_idx):
        # Initialise inputs
        self.debug = configs.debug 
        self.logger = logger 
        self.env_idx = env_idx
        self.epochs = configs.train.optimizer.epochs 
        # Set up meters and stat trackers 
        self.loss_total_meter = AverageMeter()
        self.loss_pr_meter = AverageMeter()
        self.loss_inc_meter = AverageMeter()
        self.num_triplets_meter = AverageMeter()
        self.non_zero_triplets_meter = AverageMeter()
        self.embedding_norm_meter = AverageMeter()

        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = new_environment_pickle, memory = memory)

        # Build models and init from pretrained_checkpoint
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model_frozen = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')

        self.loss_weights = nn.Parameter(torch.ones(4)) ## learnable parameters init to 1

        if configs.train.loss.incremental.name == 'MI':
            self.cp_loss = get_cp_loss()
            self.new_loss = get_new_loss()
            self.seen_loss = get_seen_loss()
            
            self.cp_loss_meter = AverageMeter()
            self.new_loss_meter = AverageMeter()
            self.seen_loss_meter = AverageMeter()

            if configs.train.uncertainty_weight:
                
                self.weight_0 = AverageMeter()
                self.weight_1 = AverageMeter()
                self.weight_2 = AverageMeter()
                self.weight_3 = AverageMeter()


        if configs.train.optimizer.name == "SGD":
            if configs.train.uncertainty_weight:
                self.optimizer = torch.optim.SGD([
                                                {"params": self.model_new.parameters()},
                                                {"params": [self.loss_weights]}  
                                                ], lr=configs.train.optimizer.lr,momentum=configs.train.optimizer.momentum,
                                                weight_decay=configs.train.optimizer.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model_new.parameters(), lr=configs.train.optimizer.lr,
                                            momentum=configs.train.optimizer.momentum,
                                            weight_decay=configs.train.optimizer.weight_decay)
        elif configs.train.optimizer.name == "Adam":
            if configs.train.uncertainty_weight:
                self.optimizer = torch.optim.Adam([{"params": self.model_new.parameters()},{"params": [self.loss_weights]}], lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model_new.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)


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
        self.loss_fn = make_pr_loss()
        self.loss_fn_inc = make_inc_loss()
        

        if configs.train.loss.incremental.name == 'EWC':
            self.fisher_matrix, self.old_parameters = self.loss_fn_inc.get_fisher_matrix(
                                                        dataloader = self.dataloader,
                                                        model = self.model_new, 
                                                        optimizer = self.optimizer, 
                                                        loss_fn = self.loss_fn
        
                                                        )
        
                


    def before_epoch(self, epoch):        
        # Reset meters
        self.loss_total_meter.reset()
        self.loss_pr_meter.reset()
        self.loss_inc_meter.reset()
        self.num_triplets_meter.reset()
        self.non_zero_triplets_meter.reset()
        self.embedding_norm_meter.reset()

        if configs.train.loss.incremental.name == 'MI':
            self.cp_loss_meter.reset()
            self.new_loss_meter.reset()
            self.seen_loss_meter.reset()
        
        # Adjust weight of incremental loss function if constraint relaxation enabled
        self.loss_fn_inc.adjust_weight(epoch)

    def training_step(self, batch_non_memories, positives_mask, negatives_mask,batch_memories,memories_pos_mask,memories_neg_mask):
        
        # Prepare batch

        if configs.model.name == 'logg3d':
            batch = batch_non_memories.to('cuda')
            memories_batch = batch_memories.to('cuda')
        else:
            batch = {x: batch_non_memories[x].to('cuda') if x!= 'coords' else batch_non_memories[x] for x in batch_non_memories}
            memories_batch = {x: batch_memories[x].to('cuda') if x!= 'coords' else batch_memories[x] for x in batch_memories}
        

        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        if n_positives == 0 or n_negatives == 0:
            # Skip a batch without positives or negatives
            print('WARNING: Skipping batch without positive or negative examples')
            return None 
        
        # Get embeddings and Loss
        self.optimizer.zero_grad()
        with torch.no_grad():
            embeddings_old,projectors_old = self.model_frozen(batch)
            embeddings_memory_old,projectors_memory_old = self.model_frozen(memories_batch)
        embeddings_memory_new,projectors_memory_new = self.model_new(memories_batch)
        embeddings_new, projectors_new = self.model_new(batch)
        
        loss_place_rec_1, num_triplets_1, non_zero_triplets_1, embedding_norm_1 = self.loss_fn(embeddings_new, positives_mask, negatives_mask)  ## Calculate the loss in the batch.
        
        loss_place_rec_2, num_triplets_2, non_zero_triplets_2, embedding_norm_2 = self.loss_fn(embeddings_memory_new, memories_pos_mask, memories_neg_mask) ## Calculate the loss in the memory.
        
        loss_place_rec = loss_place_rec_1 + loss_place_rec_2 ## Total triplet loss based domain-specific loss

        ## for tensorboard
        num_triplets = num_triplets_1 + num_triplets_2
        non_zero_triplets = non_zero_triplets_1 + non_zero_triplets_2
        embedding_norm = (embedding_norm_1 + embedding_norm_2) / 2

        
        if configs.train.loss.incremental.name == 'EWC':
            loss_incremental = self.loss_fn_inc(self.model_new, self.old_parameters, self.fisher_matrix)
        elif configs.train.loss.incremental.name == 'MI':
            loss_cp = self.cp_loss(projectors_memory_old, projectors_memory_new, memories_pos_mask,memories_neg_mask) ## MI maximization between current descriptor Zt and past descriptor
            loss_seen = self.seen_loss(projectors_memory_new,memories_pos_mask,memories_neg_mask) ##MI maximization between input X and descriptor Z Calculate on memory
            loss_new = self.new_loss(projectors_new,positives_mask,negatives_mask) ##MI maximization between input X and descriptor Z Calculate on batch
            loss_incremental = loss_cp + loss_seen + loss_new  ## add
            
            if not configs.train.uncertainty_weight:
                loss_incremental = loss_cp + loss_seen + loss_new
        else:
            loss_incremental = self.loss_fn_inc(embeddings_old, embeddings_new) ## for SA

        
        if not configs.train.uncertainty_weight: ## SA, EWC or MI(Adaptive loss weighting=False)
            loss_total = loss_place_rec + loss_incremental 
        else:## MI and Adaptive loss weighting=True 
            #TODO 
            #The Adaptive loss weighting has bugs with SGD optimizer
            loss_list = [loss_place_rec, loss_cp, loss_seen, loss_new]

            final_loss = []
            for i in range(len(self.loss_weights)):
                final_loss.append(loss_list[i] / (2 * self.loss_weights[i].pow(2)) + torch.log(self.loss_weights[i]))
            loss_total = torch.sum(torch.stack(final_loss))

        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_total_meter.update(loss_total.item())
        self.loss_pr_meter.update(loss_place_rec.item())

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

        
        
        self.num_triplets_meter.update(num_triplets)
        self.non_zero_triplets_meter.update(non_zero_triplets)
        self.embedding_norm_meter.update(embedding_norm)

        return None 

    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()

        # Dynamic Batch Expansion
        if configs.train.batch_expansion_th is not None:
            ratio_non_zeros = self.non_zero_triplets_meter.avg / self.num_triplets_meter.avg 
            if ratio_non_zeros < configs.train.batch_expansion_th:
                self.dataloader.batch_sampler.expand_batch()

        # Tensorboard plotting 
        self.logger.add_scalar(f'Step_{self.env_idx}/Total_Loss', self.loss_total_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Place_Rec_Loss', self.loss_pr_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Incremental_Loss', self.loss_inc_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Non_Zero_Triplets', self.non_zero_triplets_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Embedding_Norm', self.embedding_norm_meter.avg, epoch)
        if configs.train.uncertainty_weight and configs.train.loss.incremental.name == 'MI':
            self.logger.add_scalar(f'Step_{self.env_idx}/Seen_loss', self.seen_loss_meter.avg, epoch)
            self.logger.add_scalar(f'Step_{self.env_idx}/New_loss', self.new_loss_meter.avg, epoch)
            self.logger.add_scalar(f'Step_{self.env_idx}/Cp_loss', self.cp_loss_meter.avg, epoch)
            if configs.train.uncertainty_weight:
                self.logger.add_scalar(f'Step_{self.env_idx}/Place_Rec_Loss_weight', self.weight_0.avg, epoch)
                self.logger.add_scalar(f'Step_{self.env_idx}/Cp_loss_weight', self.weight_1.avg, epoch)
                self.logger.add_scalar(f'Step_{self.env_idx}/Seen_loss_weight', self.weight_2.avg, epoch)
                self.logger.add_scalar(f'Step_{self.env_idx}/New_loss_weight', self.weight_3.avg, epoch)
                

    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (batch, positives_mask, negatives_mask,batch_memories,memories_pos_mask,memories_neg_mask) in enumerate(self.dataloader):
                self.training_step(batch, positives_mask, negatives_mask,batch_memories,memories_pos_mask,memories_neg_mask)
            self.after_epoch(epoch)

        return self.model_new
