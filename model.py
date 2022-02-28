from torch.nn.functional import dropout
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, RGCNConv, FastRGCNConv, GATConv, global_add_pool
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GINConv
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits
import matplotlib.pyplot as plt
import torch


class Net(torch.nn.Module):
  def __init__(self, in_channels, hid_channels, out_channels):
    super(Net, self).__init__()
    # 1st type of graph layer
    self.conv1 = GCNConv(in_channels, hid_channels)
    self.conv2 = GCNConv(hid_channels, hid_channels)
    self.conv3 = GCNConv(hid_channels, out_channels)
    # 2nd type of graph layer
    self.conv4 = SAGEConv(in_channels=in_channels, out_channels=hid_channels)
    self.conv5 = SAGEConv(in_channels=hid_channels, out_channels=hid_channels)
    self.conv6 = SAGEConv(in_channels=hid_channels, out_channels=out_channels)
    # 3rd type of graph layer
    self.conv7 = GATConv(in_channels, hid_channels, heads=1, dropout=0.6)
    self.conv8 = GATConv(hid_channels*1, out_channels, concat=False, heads=8, dropout=0.6)
    self.conv9 = GATConv(hid_channels*1, hid_channels, concat=False, heads=4, dropout=0.6)


  def encode(self, x, edge_index, index = 0):
    if index == 0:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv2(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    elif index == 1:
      x = self.conv4(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv5(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv6(x, edge_index)
    elif index == 2:
      x = self.conv7(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv9(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv8(x, edge_index)
    elif index == 3:
      x = self.conv4(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv2(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv6(x, edge_index)
    elif index == 4:
      x = self.conv7(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv5(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv8(x, edge_index)
    elif index == 5:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv9(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    ####################################################
    elif index == 6:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    elif index == 7:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = self.conv3(x, edge_index)
    elif index == 8:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv2(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    elif index == 9:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      for i in range(2):
        x = self.conv2(x, edge_index)
        x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    elif index == 10:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      for i in range(3):
        x = self.conv2(x, edge_index)
        x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    elif index == 11:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      for i in range(4):
        x = self.conv2(x, edge_index)
        x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    elif index == 12:
      x = self.conv1(x, edge_index)
      x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      for i in range(10):
        x = self.conv2(x, edge_index)
        x = x.relu()
      x = dropout(x, p=0.5, training=self.training)
      x = self.conv3(x, edge_index)
    return x
      
  
  def decode(self, z, pos_edge_index, neg_edge_index):
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    return logits
  
  def decode_all(self, z):
    prob_adj = z @ z.t()
    return (prob_adj > 0).nonzero(as_tuple=False).t(), prob_adj


