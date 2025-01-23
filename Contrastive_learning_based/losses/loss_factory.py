from torchpack.utils.config import configs 
from losses.inc_loss import * 
from losses.contrastive_loss import * 


def make_inc_loss():
    inc_loss_name = configs.train.loss.incremental.name 
    if inc_loss_name == None or inc_loss_name == 'None':
        loss_fn = NoIncLoss()
    elif inc_loss_name == 'KD':
        loss_fn = KD()
    elif inc_loss_name == 'MI':
        loss_fn = MI()
    elif inc_loss_name == 'LwF':
        loss_fn = LwF()
    elif inc_loss_name == 'EWC':
        loss_fn = EWC()
    elif inc_loss_name == 'StructureAware':
        loss_fn = StructureAware()
    else:
        raise NotImplementedError(f'Unknown Loss : {inc_loss_name}')
    return loss_fn

def make_contrastive_loss():
    
    loss_fn = ContrastiveLoss(temperature=0.07)
    return loss_fn

def get_cp_loss():
    return Current_Past_MI()

def get_new_loss():
    return New_MI()

def get_seen_loss():
    return Seen_MI()
