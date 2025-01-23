import random 
import pickle 
import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
from torchpack.utils.config import configs
import copy 


class Memory:
    def __init__(self):
        self.K = 256  # 总长度
        self.max_envs = 4  # 最多环境数量
        self.num_envs = 0  # 当前环境数量
        self.env_mem_lengths = []  # 每个环境的长度列表
        self.env_mem_triplet = {0:[], 1:[], 2:[], 3:[]}
        self.train_tuples = None
        # self.env_mem = None
        
    def get_tuples(self, new_dataset_len = 0):
        tuples_dict_total = dict()
        for i in range(self.num_envs):
            train_tuples = self.env_mem_triplet[i]
            if len(train_tuples) == 0:
                continue
            tuples = copy.deepcopy(list(itertools.chain.from_iterable(train_tuples)))

            # Adjust id, positives, non_negatives to match dataset we'll be appending to 
            for t in tuples:
                t.id = t.id + new_dataset_len
                t.positives = t.positives + new_dataset_len
                t.non_negatives = t.non_negatives + new_dataset_len
                t.negatives = t.negatives + new_dataset_len
        
            tuples_dict = {t.id: t for t in tuples}
            tuples_dict_total.update(tuples_dict)
        return tuples_dict_total

    def adjust_positive_non_negative_idx(self):
        if self.num_envs == 1:
            env_replaced_idx = [[i for i in range(256)]]
        elif self.num_envs == 2:
            env_replaced_idx = [[i for i in range(128)], [i for i in range(128,256)]]
        elif self.num_envs == 3:
            env_replaced_idx = [[i for i in range(86)], [i for i in range(86,171)], [i for i in range(171,256)]]
        elif self.num_envs == 4:
            env_replaced_idx = [[i for i in range(64)], [i for i in range(64,128)], [i for i in range(128,192)], [i for i in range(192,256)]]
        env_mem = []
        for i in range(self.num_envs):
            train_tuples = self.env_mem_triplet[i]
            if len(train_tuples) == 0:
                continue
            env_tuples = list(itertools.chain.from_iterable(train_tuples))
            old_idx = [t.id for t in env_tuples]
            
            # new_idx = list(itertools.chain.from_iterable([3*x, 3*x + 1, 3*x + 2] for x in env_replaced_idx[i]))
            new_idx = list(itertools.chain.from_iterable([2*x, 2*x + 1] for x in env_replaced_idx[i]))
            old_to_new_id = {o:n for o,n in zip(old_idx, new_idx)}

            # Replace all the positives and non_negatives with new idx
            for idx, t in enumerate(env_tuples):
                positives = t.positives         ##当前t的正样本
                non_negatives = t.non_negatives ##当前t的非负样本
                negatives = t.negatives         ##当前t的非负样本
                
                new_id = old_to_new_id[t.id]
                new_positives = [old_to_new_id[p] for p in positives if p in old_to_new_id.keys()]
                new_non_negatives = [old_to_new_id[p] for p in non_negatives if p in old_to_new_id.keys()]
                new_negatives = [old_to_new_id[p] for p in negatives if p in old_to_new_id.keys()]

                t.id = new_id
                t.positives = np.sort(new_positives)
                t.non_negatives = np.sort(new_non_negatives)
                t.negatives = np.sort(new_negatives)
                env_tuples[idx] = t
            
            
            env_tuples_paired = [[env_tuples[x], env_tuples[x+1]] for x in list(range(len(env_tuples)))[::2]]
            assert len(env_replaced_idx[i]) == len(env_tuples_paired)
            for pair, replace_idx in zip(env_tuples_paired, env_replaced_idx[i]):
                reindex = replace_idx - env_replaced_idx[i][0]
                self.env_mem_triplet[i][reindex] = pair 
            env_mem = env_mem + self.env_mem_triplet[i]
        
        return env_mem
        

   
    
    @property
    def env_mem_size(self):
        if self.num_envs == 0:
            return [0]
        elif self.num_envs == 1:
            return [256]
        elif self.num_envs == 2:
            return [128,128]
        elif self.num_envs == 3:
            return [86,85,85]
        elif self.num_envs == 4:
            return [64,64,64,64]
    
    def check_and_delete(self):
        for env_idx,length in enumerate(self.env_mem_size):
            if len(self.env_mem_triplet[env_idx]) > length:
                # Memory中长度超了分配的长度
                print(len(self.env_mem_triplet[env_idx]), length , env_idx)
                self.env_mem_triplet[env_idx] = self.env_mem_triplet[env_idx][:length]
            elif len(self.env_mem_triplet[env_idx]) == length:
                print('OK')
            else:
                print('error, 当前数据长度不足')


    def update_memory(self, new_pickle, env_idx):

        self.num_envs += 1
        new_tuples = pickle.load(open(new_pickle, 'rb'))

        new_tuples_idx = list(range(len(new_tuples)))
        random.shuffle(new_tuples_idx) # Randomly shuffle the order of new tuples for selection

        num_added = 0
        selected_idx = [] # List of already selected positives; prevent double dipping!

        num_to_add = self.env_mem_size[env_idx]
        
        # Replace tuples 
        while(num_added < num_to_add):
            # Get new tuple pair to append to list 
            anchor_idx = new_tuples_idx.pop(0) #选取第0个打乱后的index，相当于随机选取
            if anchor_idx in selected_idx: # Skip if already been selected
                continue 
            anchor_tuple = new_tuples[anchor_idx]
            pair_idx_possibilities = [x for x in anchor_tuple.positives if x not in selected_idx]#将锚点元组中的所有的正样本的index取出，如果没有重复选择的话
            
            
            # Check a valid positive pair is possible 
            if len(pair_idx_possibilities) == 0:
                continue 
            
            pair_idx = random.choice(pair_idx_possibilities) ##为anchor随机选择一个正样本
            pair_tuple = new_tuples[pair_idx]##得到这个正样本对的所有信息

            

            selected_idx += [anchor_idx, pair_idx] # Prevent these being picked again

            # Get replace idx 
            if len(self.env_mem_triplet[env_idx]) < num_to_add: # Just fill up if less than K pairs in memory
                self.env_mem_triplet[env_idx].append([anchor_tuple, pair_tuple]) #如果memory中训练元组不足256对，则将选择到的锚点和正样本组成的正样本对放进去


            num_added += 1 
            if len(new_tuples_idx) == 0:
                print(f'Warning: Ran out of examples when adding memory for pickle file {new_pickle} at environment # {env_idx}')
                break 
        
        self.check_and_delete()


        self.train_tuples = self.adjust_positive_non_negative_idx()
        

if __name__ == '__main__':
    
    # 示例用法
    memory = Memory()
    new_pkl = '/media/ros/SSData/dataset/Incloud_dataset/pkl_new/Oxford_train_queries.pickle'
    env_idx = 0
    
    memory.update_memory(new_pkl,env_idx)  # 1个环境
    

    new_pkl = '/media/ros/SSData/dataset/Incloud_dataset/pkl_new/DCC_train.pickle'
    env_idx = 1
    
    
    memory.update_memory(new_pkl,env_idx)  # 1个环境
    
    tuples_dict = memory.get_tuples(0)
    print(list(tuples_dict))
    # new_pkl = '/media/ros/SSData/dataset/Incloud_dataset/pkl_new/Riverside_train.pickle'
    # env_idx = 2
    
    # memory.update_memory(new_pkl,env_idx)  # 1个环境
    

    # new_pkl = '/media/ros/SSData/dataset/Incloud_dataset/pkl_new/In-house_train_queries.pickle'
    # env_idx = 3
    

        
