from operator import mod
from torch.nn.functional import dropout
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, RGCNConv, FastRGCNConv, GATConv, global_add_pool
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GINConv
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def get_link_labels(pos_edge_index, neg_edge_index, device='cpu'):
  E = pos_edge_index.size(1) + neg_edge_index.size(1)
  link_labels = torch.zeros(E, dtype=torch.float, device=device)
  link_labels[:pos_edge_index.size(1)] = 1
  return link_labels

def train(optimizer, data, model, index=0):
  model.train()
  neg_edge_index = negative_sampling(
      edge_index=data.train_pos_edge_index,
      num_nodes=data.num_nodes,
      num_neg_samples=data.train_pos_edge_index.size(1))
  
  optimizer.zero_grad()

  z = model.encode(x=data.x, edge_index=data.train_pos_edge_index, index=index)
  link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
  link_logits = link_logits.sigmoid()
  
  link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
  loss = binary_cross_entropy(link_logits, link_labels) 
  loss.backward()
  optimizer.step()

  return loss

@torch.no_grad()
def test(data, model, index=0):
  model.eval()
  prefs = []
  f_score = []
  aupr_score = []

  for prefix in ['val', 'test']:
    pos_edge_index = data[f'{prefix}_pos_edge_index']
    neg_edge_index = data[f'{prefix}_neg_edge_index']

    z = model.encode(x=data.x, edge_index=data.train_pos_edge_index, index=index)
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    prefs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    aupr_score.append(average_precision_score(link_labels.cpu(), link_probs.cpu()))
    # f_score.append(accuracy_score(np.array(link_labels.cpu()), np.array(link_probs.cpu())))
    f_score.append(0)

  return prefs, aupr_score, f_score

def plot_result(train_loss_list, val_acc_list, test_acc_list):
  plt.figure(figsize=(30, 5))
  plt.subplot(131)
  plt.grid()
  plt.plot(train_loss_list)
  plt.title('train loss')
  plt.subplot(132)
  plt.grid()
  plt.plot(val_acc_list)
  plt.title('val acc')
  plt.subplot(133)
  plt.plot(test_acc_list)
  plt.plot('test acc')
  plt.show()

def train_model(model, optimizer, data, num_epochs, index=0):
  best_val_pref = test_pref = 0
  train_loss_list = []
  val_auc_list = []
  test_auc_list = []
  val_aupr_list = []
  test_aupr_list = []
  val_fscore_list = []
  test_fscore_list = []

  
  for epoch in range(0, num_epochs):
    train_loss = train(optimizer=optimizer, data=data, model=model, index=index)
    val_pref , test_pref, val_aupr, test_aupr, val_fscore, test_fscore = test(data=data, model=model, index=index)
    log = 'Epoch: {:03d}, Loss: {:04f}, Val: {:04f}, Test: {:04f}'
    train_loss_list.append(train_loss.detach().numpy())

    val_auc_list.append(val_pref)
    test_auc_list.append(test_pref) 
    val_aupr_list.append(val_aupr)
    test_aupr_list.append(test_aupr)
    val_fscore_list.append(val_fscore)
    test_fscore_list.append(test_fscore)

    if epoch % 100 == 0:
      print(log.format(epoch, train_loss, val_pref, test_pref))
  plot_result(train_loss_list, val_auc_list, test_auc_list)


def train_model_on_folds(folds, num_epochs, index=0, in_channels=None, hid_channels=128, out_channels=64, device=None, model_class=None, lr=0.0005, verbose=1):

  train_loss = []
  val_acc = []
  test_acc = []
  val_aupr = []
  test_aupr = []
  val_fscore = []
  test_fscore = []

  for fold in folds:
    model = model_class(in_channels=in_channels, hid_channels=hid_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr=0.0005)

    test_pref = 0
    train_loss_list = []
    val_acc_list = []
    test_acc_list = []

    val_aupr_list = []
    test_aupr_list = []
    val_fscore_list = []
    test_fscore_list = []
    
    for epoch in range(0, num_epochs):
      loss = train(optimizer=optimizer, data=fold, model=model, index=index)
      (val_pref , test_pref), (v_aupr, t_aupr), (v_fscore, t_fscore) = test(data=fold, model=model, index=index)
      train_loss_list.append(loss.detach().numpy())
      val_acc_list.append(val_pref)
      test_acc_list.append(test_pref)
      val_aupr_list.append(v_aupr)
      test_aupr_list.append(t_aupr)
      val_fscore_list.append(v_fscore)
      test_fscore_list.append(t_fscore) 

    train_loss.append(train_loss_list)
    val_acc.append(val_acc_list)
    test_acc.append(test_acc_list)
    val_aupr.append(val_aupr_list)
    test_aupr.append(test_aupr_list)
    val_fscore.append(val_fscore_list)
    test_fscore.append(test_fscore_list)

  train_loss = np.array(train_loss)
  val_acc = np.array(val_acc)
  test_acc = np.array(test_acc)

  train_loss = np.average(train_loss, axis=0)
  val_acc = np.average(val_acc, axis=0)
  test_acc = np.average(test_acc, axis=0)
  val_aupr = np.average(val_aupr, axis=0)
  test_aupr = np.average(test_aupr, axis=0)
  val_fscore = np.average(val_fscore, axis=0)
  test_fscore = np.average(test_fscore, axis=0)


  print_train_result(train_loss, val_acc, test_acc, verbose=verbose)
  plot_result(train_loss, val_acc, test_acc)

  return train_loss, val_acc, test_acc, model, val_aupr, test_aupr, val_fscore, test_fscore


def device_finder():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return device


def print_train_result(train_loss, val_acc, test_acc, verbose=1):
  size = train_loss.shape[0]
  for epoch in range(size):
    if size == 100:
      t = 10
    elif size == 1000:
      t = 100
    else:
      t = 10
    if epoch % t == 0:
      loss = train_loss[epoch]
      val = val_acc[epoch]
      test = test_acc[epoch]
      log = 'Epoch: {:03d}, Train Loss: {:04f}, Val Auc: {:04f}, Test Auc: {:04f}'
      if verbose == 1:
        print(log.format(epoch, loss, val, test))
  
  if verbose == 0:
    print(log.format(epoch, loss, val, test))

def predict_edges(data, prob_adj, adj_prob_threshold=0.9):
  a = (prob_adj > adj_prob_threshold).nonzero().t()

  correct = 0
  c_edges = np.array(data.edge_index.T)
  for edge in a.T:
      edge = np.array(edge)
      if edge in c_edges:
          correct += 1

  return c_edges

def plot_layers_curve(model_layers, loss, val, test):
  plt.figure(figsize=(18, 5))

  plt.subplot(131)
  plt.scatter(model_layers, loss)
  plt.plot(model_layers, loss)
  plt.title('train loss')
  plt.xlabel('number of hidden layers')
  plt.ylabel('loss')

  plt.subplot(132)
  plt.scatter(model_layers, val)
  plt.plot(model_layers, val)
  plt.title('val acc')
  plt.xlabel('number of hidden layers')
  plt.ylabel('ROC AUC Score')

  plt.subplot(133)
  plt.scatter(model_layers, test)
  plt.plot(model_layers, test)
  plt.title('test acc')
  plt.xlabel('number of hidden layers')
  plt.ylabel('ROC AUC Score')

  plt.show()

def plot_comparison(roc_auc_score):
    method_names = ['GTB [2019]', 'AuDNNsynergy [2021]', 'DComG']
    scores = [0.949, 0.925, roc_auc_score]

    plt.figure(figsize=(6, 6))

    d = {'models': method_names, 'values': scores}
    df = pd.DataFrame(d, columns=['models', 'values'])
    plots = sns.barplot(x='models', y='values', data=df, color='gray')

    width = 0.6

    plt.title('AUC Test Score')
    for i, bar in enumerate(plots.patches):
        plots.annotate(format(bar.get_height(), '.3f'),
        (bar.get_x() + bar.get_width() / 2,
        bar.get_height()), ha='center', va='center', 
        size=15, xytext=(-12, 10),
        textcoords='offset points')
        bar.set_width(width)

    plt.ylim(0.92, 0.99)
    plt.grid()
    plt.show()