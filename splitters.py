import torch
import math
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def k_fold_edge_split(data, val_ratio: float = 0.1, test_ratio: float = 0.1, folds=5, only_upper_triangular_portion: bool = False):
    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_attr = None

    # Return upper triangular portion.
    if only_upper_triangular_portion == True:
        mask = row < col
        row, col = row[mask], col[mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    size = n_v + n_t
    
    data_folds = []

    for k in range(folds):
        row, col = data.edge_index
        fold = Data(x=data.x)
        
        # Positive edges
        pos_perm = torch.randperm(row.size(0))
        row, col = row[pos_perm], col[pos_perm]
        if edge_attr is not None:
            edge_attr = edge_attr[pos_perm]

        r_t, r_c = row[size*k:size*(k+1)], col[size*k:size*(k+1)]

        r, c = r_t[:n_v], r_c[:n_v]
        fold.val_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = r_t[n_v:n_v + n_t], r_c[n_v:n_v + n_t]
        fold.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = torch.concat([row[:size*k], row[size*(k+1):]], dim=0), torch.concat([col[:size*k], col[size*(k+1):]], dim=0)
        fold.train_pos_edge_index = torch.stack([r, c], dim=0)

        if only_upper_triangular_portion == True:
            fold.train_pos_edge_index = to_undirected(fold.train_pos_edge_index)

        # Negative edges.
        row, col = data.edge_index

        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        neg_perm = torch.randperm(neg_row.size(0))[size*k:size*(k+1)]
        neg_row, neg_col = neg_row[neg_perm], neg_col[neg_perm]

        neg_adj_mask[neg_row, neg_col] = 0
        fold.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        fold.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        fold.test_neg_edge_index = torch.stack([row, col], dim=0)

        data_folds.append(fold)
    return data_folds

def train_test_split_edges(new_data, val_ratio: float = 0.1,
                           test_ratio: float = 0.1):
    assert 'batch' not in new_data  # No batch-mode.

    num_nodes = new_data.num_nodes
    row, col = new_data.edge_index
    edge_attr = new_data.edge_attr

    data = Data(x=new_data.x, edge_index=new_data.edge_index)
    data.edge_index = data.edge_attr = None


    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
