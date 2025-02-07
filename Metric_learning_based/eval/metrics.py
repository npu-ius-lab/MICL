import numpy as np 
import pandas as pd 

class IncrementalTracker:
    def __init__(self):
        self.most_recent = {}
        self.greatest_past = {}
        self.start_indexes = {} # Recall@1 for latest 
        self.seen_envs = []


    def update(self, update_dict, env_idx):
        for k,v in update_dict.items():
            if k not in self.seen_envs:
                self.seen_envs.append(k)
                self.most_recent[k] = v  
                self.greatest_past[k] = np.nan 
                self.start_indexes[k] = env_idx 
            else:
                self.greatest_past[k] = self.most_recent[k] if np.isnan(self.greatest_past[k]) else max(self.greatest_past[k], v)
                self.most_recent[k] = v 

    def get_results(self):
        # Get recall and forgetting
        results = {}
        for k in self.start_indexes:
            results[k] = {}
            results[k]['Recall@1'] = self.most_recent[k]
            if k in self.greatest_past:
                results[k]['Forgetting'] = self.greatest_past[k] - self.most_recent[k]
            else:
                results[k]['Forgetting'] = np.nan
        
        # Merge
        results_merged = {} 
        for v in self.start_indexes.values():
            merge_keys = [k for k in self.start_indexes if self.start_indexes[k] == v] # Get keys which should be merged 
            new_key = '/'.join(merge_keys) # Get new key 
            merged_recall = np.mean([results[m]['Recall@1'] for m in merge_keys])
            merged_forgetting = np.mean([results[m]['Forgetting'] for m in merge_keys])

            results_merged[new_key] = {'Recall@1': merged_recall, 'Forgetting': merged_forgetting}

        # Print 
        results_final = pd.DataFrame(columns = ['Recall@1', 'Forgetting'])
        for k in results_merged:
            results_final.loc[k] = [results_merged[k]['Recall@1'], results_merged[k]['Forgetting']]
        results_final.loc['Average'] = results_final.mean(0)
        return results_final 

