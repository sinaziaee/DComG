from utils.dataset_loader import load_dataframes
from model import Net
import warnings

import torch
from torch_geometric.data import Data
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


def create_graph_data(df, all_edges):
    r_size, c_size = df.shape

    x_data = df.iloc[:, :(c_size-2)]
    x_data = np.array(x_data, dtype=np.float32)

    nodes_data_list = list()
    nodes_data_dict = dict()
    reverse_node_data_dict = dict()


    count = 0
    for x in df.values[:, (c_size-2):(c_size-1)]:
        nodes_data_dict[str(x.squeeze())] = count
        reverse_node_data_dict[count] = str(x.squeeze())
        count+=1
        nodes_data_list.append(str(x.squeeze()))

    edges_data = list()
    for edge in all_edges:
        if edge[0] in nodes_data_list and edge[1] in nodes_data_list:
            edges_data.append([nodes_data_dict[edge[0]], nodes_data_dict[edge[1]]])
    nodes_data = list(nodes_data_dict.values())
    nodes_data = torch.from_numpy(np.array(nodes_data))
    edges_data = torch.from_numpy(np.array(edges_data))
    x_data = torch.from_numpy(np.array(x_data))

    data = Data(x=x_data, edge_index=edges_data.T)
    return data, reverse_node_data_dict


def create_graph_data_with_different_features(with_reverse=False):
    # final_dt_df -> onehot drug-target     : 760 * 790
    # final_w2v_df -> word2vec              : 614 * 200
    # final_nv_df -> node2vec               : 752 * 128
    # final_fin_df -> drug fingerprint      : 627 * 167
    # final_in_df -> drug indication        : 383 * 1513
    # final_se_df -> drug side effect       : 389 * 3256
    # all_df -> all drug combinations       : 2716 * 2
    final_dt_df, final_w2v_df, final_nv_df, final_fin_df, final_in_df, final_se_df, all_df = load_dataframes()

    all_edges = []
    for edge in all_df.values:
        if list(edge) not in all_edges and [edge[1], edge[0]] not in all_edges:
            all_edges.append(list(edge))
            all_edges.append([edge[1], edge[0]])

    df_list = [final_dt_df, final_w2v_df, final_nv_df, final_fin_df, final_in_df, final_se_df]
    data_list = []
    reverse_node_dict_list = []
    for df in df_list:
        data, reverse_node_data_dict = create_graph_data(df, all_edges)
        data_list.append(data)
        reverse_node_dict_list.append(reverse_node_data_dict)
    
    if with_reverse == True:
        return data_list, reverse_node_dict_list
    else:
        return data_list
