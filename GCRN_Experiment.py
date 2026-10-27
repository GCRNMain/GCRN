import h5py
import scipy.io as sio
import numpy as np
import time
import pandas as pd

from GCRN import GSCNCenter
from GCRN import functionTree
from GCRN import MHFS_DataHandle


def experiment_GCRN_center():
    mat_names = ["D:\\Python\\9GCRN\\datasets\\Tree\\train\\BridgesTrain.mat"]
    save_names = ["D:\\Python\\9GCRN\\result\\1GCRN\\BridgesTrainCenter_GCRN.mat"]
    for name_i in range(0, len(mat_names)):
        mat_name = mat_names[name_i]
        save_name = save_names[name_i]

        if(name_i == int(4)):
            with h5py.File(mat_name, 'r') as f:
                tree = np.array(f['tree']).T
                data_array = np.array(f['data_array']).T
        else:
            data_array = sio.loadmat(mat_name)
            tree = data_array['tree']
            data_array = data_array['data_array']

        data_array = MHFS_DataHandle.MHFS_DataHandle(data_array)
        [data_array_Xs, data_array_Ys] = functionTree.creatSubTablezh_RL(data_array, tree)

        start_time = time.time()
        # 1.center_arr 
        center_arr = {}
        for node_id, X_i_data in data_array_Xs.items():
            cluster_X = X_i_data
            if(cluster_X.shape[0] == 0):
                continue
            _, idx1 = np.unique(data_array_Ys[node_id], return_index=True)
            idx_sorted = np.sort(idx1)
            data_array_i_labels = data_array_Ys[node_id][idx_sorted]
            cluster_K = len(data_array_i_labels)

            idx, center, _, _, _ = GSCNCenter.train_stable_gcn(cluster_X, cluster_K, epochs=10, alpha=0.1)

            labels_col = data_array_i_labels.reshape(-1, 1).astype(int)
            center_and_label = np.hstack([center, labels_col])
            center_arr[node_id] = center_and_label
            pass

        data_array_X_Root_sample = list(data_array_Xs.values())[-1].shape[0]
        trulabels = np.unique(data_array[:, -1]).astype(int)
        center_root = list(center_arr.values())[-1]

        # for
        # 2.
        while True:
            current_labels = center_root[:, -1].astype(int)
            #
            non_true_mask = ~np.isin(current_labels, trulabels)
            non_true_labels = current_labels[non_true_mask]
            if len(non_true_labels) == 0:
                break

            new_rows = []
            for idx in range(len(center_root)):
                center_i_label = int(center_root[idx, -1])
                if center_i_label in trulabels:
                    new_rows.append(center_root[idx:idx + 1])
                else:  #
                    ###
                    center_no_ture_label = center_root[idx:idx + 1]
                    sub_centers = center_arr[center_i_label]

                    data_array_X_sub_sample = data_array_Xs[center_i_label].shape[0]
                    ratio = data_array_X_sub_sample / data_array_X_Root_sample

                    sub_feat = sub_centers[:, :-1]
                    center_feat = center_no_ture_label[:, :-1]
                    adjusted_feat = sub_feat * ratio + center_feat
                    sub_label = sub_centers[:, -1:]

                    adjusted_sub = np.hstack([adjusted_feat, sub_label])
                    new_rows.append(adjusted_sub)
                    pass

            # 
            center_root = np.vstack(new_rows)
            pass

        end_time = time.time()
        elapsed_time = end_time - start_time

        sio.savemat(save_name, {'center': center_root, 'vCenterTime': elapsed_time})
        pass


if __name__ == "__main__":
    experiment_GCRN_center()

    pass
