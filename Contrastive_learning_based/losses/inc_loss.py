import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchpack.utils.config import configs 
from tqdm import tqdm 

class NoIncLoss:
    def __init__(self):
        pass

    def adjust_weight(self, epoch):
        pass
    
    def __call__(self, *args, **kwargs):
        return torch.tensor(0, dtype = float, device = 'cuda')

class KD:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight    # λ_init
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 
        self.t = 0.1

        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())    # w(γ)


    def adjust_weight(self, epoch):
        if configs.train.loss.incremental.adjust_weight:
            self.weight = self.orig_weight * self.weight_factors[epoch - 1]
        else:
            pass 
    
    def __call__(self, projectors_memories_frozen, projectors_memories):
        with torch.no_grad():
            previous_similarity = torch.einsum('nc,cm->nm', [projectors_memories_frozen, projectors_memories_frozen.T])
            logits_mask = torch.scatter(torch.ones_like(previous_similarity),1,torch.arange(previous_similarity.size(0)).view(-1, 1).cuda(),0)
            q = torch.softmax(previous_similarity*logits_mask / self.t, dim=1)
        current_similarity = torch.einsum('nc,cm->nm', [projectors_memories, projectors_memories.T])
        log_p = torch.log_softmax(current_similarity*logits_mask / self.t, dim=1)
        loss_distill = self.weight * torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return loss_distill
    



class New_MI:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight 
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 
        self.temperature = 0.07
        self.criterion = nn.CrossEntropyLoss().cuda()
        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())


    def adjust_weight(self, epoch):
        pass
    

    def __call__(self, current_new_rep,pos_mask ,neg_mask):
        batch_size = current_new_rep.shape[0]
        
        total_loss = 0
        for i in range(batch_size):
            query_embedding = current_new_rep[i].unsqueeze(0)
            positive_i = torch.nonzero(pos_mask[i]).squeeze() 
            negative_i = torch.nonzero(neg_mask[i]).squeeze() 
            if positive_i.numel() == 0 or negative_i.numel() == 0:
                continue
            positive_current_rep = current_new_rep[positive_i,:]
            if positive_current_rep.dim() == 1:
                positive_current_rep = torch.unsqueeze(positive_current_rep, 0)
            l_pos = torch.exp(torch.einsum('bi,bi->b', [query_embedding, positive_current_rep]).unsqueeze(1) / self.temperature)
            l_pos = l_pos.mean()
            negative_embedding_new = current_new_rep[negative_i,:]
            if negative_embedding_new.dim() == 1:
                negative_embedding_new = torch.unsqueeze(negative_embedding_new, 0)
            l_neg = torch.exp(torch.einsum('bi,bi->b', [query_embedding, negative_embedding_new]).unsqueeze(1) / self.temperature)
            l_neg = l_neg.sum()
            if (l_pos / (l_pos + l_neg)) == 0:
                print('error')
            loss_item = torch.log(l_pos / (l_pos + l_neg))
            total_loss += loss_item
        loss_incremental = -total_loss / batch_size

        
        return loss_incremental



class Seen_MI:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight 
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 
        self.temperature = 0.07
        self.criterion = nn.CrossEntropyLoss().cuda()
        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())


    def adjust_weight(self, epoch):
        pass
    

    def __call__(self, mem_rep ,memories_pos_mask ,memories_neg_mask):
        batch_size = mem_rep.shape[0]
        
        total_loss = 0
        for i in range(batch_size):
            query_embedding = mem_rep[i].unsqueeze(0)
            positive_i = torch.nonzero(memories_pos_mask[i]).squeeze() 
            negative_i = torch.nonzero(memories_neg_mask[i]).squeeze()
            if positive_i.numel() == 0 or negative_i.numel() == 0:
                continue
            positive_mem_rep = mem_rep[positive_i,:]
            if positive_mem_rep.dim() == 1:
                positive_mem_rep = torch.unsqueeze(positive_mem_rep, 0)
                
            l_pos = torch.exp(torch.einsum('bi,bi->b', [query_embedding, positive_mem_rep]).unsqueeze(1) / self.temperature)
            l_pos = l_pos.mean()
            negative_embedding_mem = mem_rep[negative_i,:]
            if negative_embedding_mem.dim() == 1:
                negative_embedding_mem = torch.unsqueeze(negative_embedding_mem, 0)
            l_neg = torch.exp(torch.einsum('bi,bi->b', [query_embedding, negative_embedding_mem]).unsqueeze(1) / self.temperature)
            l_neg = l_neg.sum()
            loss_item = torch.log(l_pos / (l_pos + l_neg))
            total_loss += loss_item
        loss_incremental = -total_loss / batch_size

        
        return loss_incremental





class Current_Past_MI:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight 
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 
        self.temperature = 0.07
        self.criterion = nn.CrossEntropyLoss().cuda()
        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())


    def adjust_weight(self, epoch):
        pass 


    def __call__(self, old_mem_rep, new_mem_rep ,memories_pos_mask ,memories_neg_mask):
        batch_size = new_mem_rep.shape[0]
        

        total_loss = 0
        for i in range(batch_size):
            query_embedding = new_mem_rep[i].unsqueeze(0)
            positive_i = torch.nonzero(memories_pos_mask[i]).squeeze() 
            negative_i = torch.nonzero(memories_neg_mask[i]).squeeze() 

            if positive_i.numel() == 0 or negative_i.numel() == 0:
                continue
            positive_embedding_old = old_mem_rep[positive_i,:]
            positive_embedding_new = new_mem_rep[positive_i,:]
            if positive_embedding_old.dim() == 1:
                positive_embedding_old = torch.unsqueeze(positive_embedding_old, 0)
                positive_embedding_new = torch.unsqueeze(positive_embedding_new, 0)
            l_pos = torch.exp(torch.einsum('bi,bi->b', [query_embedding, positive_embedding_old]).unsqueeze(1) / self.temperature)
            l_pos = l_pos.mean()

            
            negative_embedding_old = old_mem_rep[negative_i,:]
            negative_embedding_new = new_mem_rep[negative_i,:]
            if negative_embedding_old.dim() == 1:
                negative_embedding_old = torch.unsqueeze(negative_embedding_old, 0)
                negative_embedding_new = torch.unsqueeze(negative_embedding_new, 0)

            l_neg = torch.exp(torch.einsum('bi,bi->b', [query_embedding, negative_embedding_old]).unsqueeze(1) / self.temperature)
            l_neg = l_neg.sum()
            loss_item = torch.log(l_pos / (l_pos + l_neg))
            total_loss += loss_item
        loss_incremental = -total_loss / batch_size

        
        return loss_incremental
    

class LwF:
    def __init__(self):
        self.weight = configs.train.loss.incremental.weight
        self.temperature = 2 
    
    def adjust_weight(self, epoch):
        pass 

    def __call__(self, old_rep, new_rep):
        log_p = torch.log_softmax(new_rep / self.temperature, dim=1)
        q = torch.softmax(old_rep / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        loss_incremental = self.weight * res
        return loss_incremental

class StructureAware:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight 
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 

        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())


    def adjust_weight(self, epoch):
        if configs.train.loss.incremental.adjust_weight:
            self.weight = self.orig_weight * self.weight_factors[epoch - 1]
        else:
            pass 

    def __call__(self, old_rep, new_rep):
        with torch.no_grad():
            old_vec = old_rep.unsqueeze(0) - old_rep.unsqueeze(1) # B x D x D
            norm_old_vec = F.normalize(old_vec, p = 2, dim = 2)
            old_angles = torch.bmm(norm_old_vec, norm_old_vec.transpose(1,2)).view(-1)

        new_vec = new_rep.unsqueeze(0) - new_rep.unsqueeze(1)
        norm_new_vec = F.normalize(new_vec, p = 2, dim = 2)
        new_angles = torch.bmm(norm_new_vec, norm_new_vec.transpose(1,2)).view(-1)

        loss_incremental = F.smooth_l1_loss(new_angles, old_angles, reduction = 'none')
        loss_incremental = F.relu(loss_incremental - self.margin)

        # Remove 0 terms from loss which emerge due to margin 
        # Only do if there are any terms where inc. loss is not zero
        if torch.any(loss_incremental > 0):
            loss_incremental = loss_incremental[loss_incremental > 0]

        loss_incremental = self.weight * loss_incremental.mean()
        return loss_incremental

class EWC:
    def __init__(self):
        self.weight = configs.train.loss.incremental.weight

    def adjust_weight(self, epoch):
        pass 

    def get_fisher_matrix(self, model, dataloader, optimizer, loss_fn):
        fisher = {n: torch.zeros(p.shape).to('cuda') for n,p in model.named_parameters() if p.requires_grad}
        pbar = tqdm(desc = 'Getting Fisher Matrix', total = len(dataloader.dataset))
        for batch, positives_mask, negatives_mask in dataloader:
            batch = {e: batch[e].to('cuda') if e!= 'coords' else batch[e] for e in batch}
            embeddings = model(batch)
            loss, _, _, _ = loss_fn(embeddings, positives_mask, negatives_mask)
            optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(embeddings)
            pbar.update(len(positives_mask))
        pbar.close()
        # Apply mean across all samples
        n_samples = len(dataloader.dataset)
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        old_parameters = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        return fisher, old_parameters
        

    def __call__(self, model_new, old_parameters, fisher_matrix):
        loss_incremental = torch.tensor(0, device = 'cuda', dtype = float)
        for n, new_param in model_new.named_parameters():
            if n in fisher_matrix.keys():
                old_param = old_parameters[n]
                loss_incremental += torch.sum(fisher_matrix[n] * (new_param - old_param).pow(2))
        loss_incremental = self.weight * loss_incremental
        return loss_incremental


class MI:
    def __init__(self):
        self.seen = Seen_MI()
        self.new = New_MI()
        self.cp = Current_Past_MI()
        
        self.orig_weight = configs.train.loss.incremental.weight 
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 
        self.temperature = 0.07
        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())


    def adjust_weight(self, epoch):
        pass

    def __call__(self, old_mem_rep, new_mem_rep, memories_pos_mask, memories_neg_mask, current_data_rep, current_pos_mask, current_neg_mask):
        loss_seen = self.seen(new_mem_rep,memories_pos_mask,memories_neg_mask)
        loss_new = self.new(current_data_rep, current_pos_mask, current_neg_mask)
        loss_cp = self.cp(old_mem_rep, new_mem_rep, memories_pos_mask, memories_neg_mask)
        

        loss_incremental = loss_cp + loss_seen + loss_new
        return loss_incremental