# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
from datasets.oxford import TrainingTuple
# Import test set boundaries
from generating_queries.Oxford.generate_test import P1, P2, P3, P4, check_in_test_set

# Test set boundaries
P = [P1, P2, P3, P4]

FILENAME = "pointcloud_locations_20m_10overlap.csv"
POINTCLOUD_FOLS = "pointcloud_20m_10overlap"


def construct_query_dict(df_centroids, save_folder, filename, ind_nn_r, ind_r_r):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    all_indexes = np.arange(len(df_centroids))
    for anchor_ndx in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']]) ##计算当前anchor的位置
        query = df_centroids.iloc[anchor_ndx]["file"]##当前anchor对应的submap文件路径
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]##当前anchor对应的submap的文件名
        # assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])##当前anchor对应的submap的时间戳

        positives = ind_nn[anchor_ndx]###所有子图中，与当前anchor距离小于10m的所有子图的index(正样本)
        non_negatives = ind_r[anchor_ndx]###所有子图中，与当前anchor距离小于50m的所有子图的index(非负样本)

        positives = positives[positives != anchor_ndx]##排除自己
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Get negatives indexes
        negatives = np.setdiff1d(all_indexes, non_negatives)
        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int], negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, negatives=negatives, position=anchor_pos)

    file_path = os.path.join(save_folder, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--pos_thresh', type = int, default = 10, help = 'Threshold for positive examples')
    parser.add_argument('--neg_thresh', type = int, default = 50, help = 'Threshold for negative examples')
    parser.add_argument('--file_extension', type = str, default = '.bin', help = 'File extension expected')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save pickle files to')
    args = parser.parse_args()

    # Check dataset root exists, make save dir if doesn't exist
    print('Dataset root oxford : {}'.format(args.dataset_root))
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    folders = sorted(os.listdir(base_path))

    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
    
    for folder in tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        df_locations['timestamp'] = base_path + '/' + folder + '/' + POINTCLOUD_FOLS + '/' + df_locations['timestamp'].astype(str) + args.file_extension
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                # df_test = df_test.append(row, ignore_index=True)
                df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
            else:
                # df_train = df_train.append(row, ignore_index=True)
                df_train = pd.concat([df_train, row.to_frame().T], ignore_index=True)
                


    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, args.save_folder, "Oxford_train_queries.pickle", args.pos_thresh, args.neg_thresh)
    construct_query_dict(df_test, args.save_folder, "Oxford_test_queries.pickle", args.pos_thresh, args.neg_thresh)
