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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.data import Data



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

def train_model_on_folds_and_decode_all(folds, num_epochs, data=None, index=0, in_channels=None, hid_channels=128, out_channels=64, device=None, model_class=None, lr=0.0005, verbose=1):

  train_loss = []
  val_auc = []
  test_auc = []
  prob_adj_list = []

  for fold in folds:
    model = model_class(in_channels=in_channels, hid_channels=hid_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr=0.0005)

    test_pref = 0
    train_loss_list = []
    val_acc_list = []
    test_acc_list = []

    
    for epoch in range(0, num_epochs):
      loss = train(optimizer=optimizer, data=fold, model=model, index=index)
      (val_pref , test_pref), (v_aupr, t_aupr), (v_fscore, t_fscore) = test(data=fold, model=model, index=index)
      train_loss_list.append(loss.detach().numpy())
      val_acc_list.append(val_pref)
      test_acc_list.append(test_pref)
    
    z = model.encode(data.x, data.edge_index, index=index)
    final_edge_index, prob_adj = model.decode_all(z)
    prob_adj_list.append(prob_adj)

    train_loss.append(train_loss_list)
    val_auc.append(val_acc_list)
    test_auc.append(test_acc_list)

  train_loss = np.array(train_loss)
  val_auc = np.array(val_auc)
  test_auc = np.array(test_auc)

  train_loss = np.average(train_loss, axis=0)
  val_auc = np.average(val_auc, axis=0)
  test_auc = np.average(test_auc, axis=0)


  print_train_result(train_loss, val_auc, test_auc, verbose=verbose)
  plot_result(train_loss, val_auc, test_auc)

  

  return train_loss, val_auc, test_auc, model, prob_adj_list

def predict_top_edges(data, prob_adj):
  scaler = MinMaxScaler()
  prob_adj = prob_adj.detach().numpy()
  prob_adj = scaler.fit_transform(prob_adj)
  threshold = 0.95
  a = np.array(list((prob_adj > threshold).nonzero())).T
  b = []
  for e in a:
      b.append(list(e))

  found = 0
  c_edges = np.array(data.edge_index.T)

  all_edges = []

  for e in c_edges:
      all_edges.append(list(e))

  for edge in b:
      if edge in all_edges:
          found += 1

  all_0 = []
  all_1 = []
  all_2 = []

  for i in range(prob_adj.shape[0]):
      for j in range(prob_adj.shape[1]):
          if i == j:
              prob_adj[i, j] = 0
          all_0.append(i)
          all_1.append(j)
          all_2.append(prob_adj[i, j])

  all_0 = np.array(all_0)
  all_1 = np.array(all_1)
  all_2 = np.array(all_2)

  all_0 = np.reshape(all_0, (len(all_0), 1))
  all_1 = np.reshape(all_1, (len(all_1), 1))
  all_2 = np.reshape(all_2, (len(all_2), 1))

  all = pd.DataFrame(np.concatenate([all_0, all_1], axis=1), columns=['alias1', 'alias2'])
  all['score'] = all_2
  all = all.sort_values('score', ascending=False)

  high_score_combinations = []
  for row in all.values:
      if row[2] >= 1:
          high_score_combinations.append(list(np.array(row[:2], dtype=np.int16)))
  
  new_predicted_edges = []
  for edge in high_score_combinations:
      if edge not in all_edges:
          new_predicted_edges.append(edge)

  return high_score_combinations, new_predicted_edges

def predict_new_edges(all_prob_adj_list, data):
  all_pairs = []
  for i in range(len(all_prob_adj_list)):
      for j in range(len(all_prob_adj_list[i])):
          for k in range(len(all_prob_adj_list[i][j])):
              prob_adj = all_prob_adj_list[i][j][k]
              high, pairs = predict_top_edges(data, prob_adj)
              all_pairs.append(pairs)
              print(len(high), len(pairs))
  all_new_pairs = []
  temp_edges = all_pairs[0]
  count = 0
  for edge in temp_edges:
      t = 0
      for i in range(len(all_pairs)):
          e_list = all_pairs[i]
          if edge in e_list:
              t += 1
      if t == len(all_pairs):
          count += 1
          all_new_pairs.append(edge)
  return all_new_pairs

def plotter(scores, method_names, colors):
  plt.figure(figsize=(30, 6))

  d = {'models': method_names, 'values': scores}
  df = pd.DataFrame(d, columns=['models', 'values'])
  plots = sns.barplot(x='models', y='values', data=df, color='gray')

  width = 0.4

  plt.title('AUC Test Score')
  for i, bar in enumerate(plots.patches):
      plots.annotate(format(bar.get_height(), '.3f'),
      (bar.get_x() + bar.get_width() / 2,
      bar.get_height()), ha='center', va='center', 
      size=15, xytext=(-12, 10),
      textcoords='offset points')
      bar.set_width(width)
      bar.set_color(colors[int(i / 6)])
  
  for item in plots.get_xticklabels():
      item.set_rotation(-45)

  plt.ylim(0.7, 0.99)
  plt.grid()
  plt.show()

def all_features_graph_data(final_in_df, df_list, all_df):
    x_in = final_in_df.iloc[:, :128]
    x_in = np.array(x_in, dtype=np.float32)
    nodes_list = list()
    nodes_dict = dict()
    reverse_nodes_dict = dict()
    ###################################################
    drug_names = []
    for drug in list(final_in_df.drugs):
        count = 0
        for df in df_list:
            if drug in list(df.drugs):
                count += 1
        if count == len(df_list):
            drug_names.append(drug)

    count = 0
    for d in final_in_df.drugs.values:
        if d in drug_names:
            nodes_dict[d] = count
            reverse_nodes_dict[count] = d
            count+=1
            nodes_list.append(d)
    # print(len(drug_names))
    ##############################################
    str_edges = []
    for row in all_df.values:
        d1 = row[0]
        d2 = row[1]
        edge = [d1, d2]
        if edge not in str_edges and [edge[1], edge[0]] not in str_edges and edge[0] in drug_names and edge[1] in drug_names:
            str_edges.append(list(edge))
            str_edges.append([edge[1], edge[0]])  
    # print(len(str_edges))  

    edges = []
    for edge in str_edges:
        d1 = nodes_dict[edge[0]]
        d2 = nodes_dict[edge[1]]
        edges.append([d1, d2])
    # print(len(edges))
    edges = torch.from_numpy(np.array(edges))
    # print(edges)
    ##########################################################
    def check_all(drug, drug_names):
        if drug in drug_names:
            return True
        else:
            return False

    drug_numbers = []
    drug_features = []

    for drug in list(nodes_dict.keys()):
        features = []
        if check_all(drug, drug_names):
            for i, df in enumerate(df_list):
                # print(df.shape)
                a = df[df.drugs == drug].iloc[:, :(df.shape[1]-2)].values.squeeze()
                a = a[:a.shape[0]]
                features.append(np.array(a, dtype=np.float32))
            drug_features.append(features)
            drug_numbers.append(drug)

    # print(len(drug_features))

    all_drug_features = []
    for i in range(len(drug_features)):
        l = []
        for j in range(len(drug_features[i])):
            z = drug_features[i][j]
            for k in z:
                l.append(k)
        all_drug_features.append(np.array(l, dtype=np.float32))
    all_drug_features = np.array(all_drug_features)
    all_drug_features = torch.from_numpy(all_drug_features)

    data_all = Data(x=all_drug_features, edge_index=edges.T)
    return data_all, reverse_nodes_dict

import pathlib
import os
################################################################
def create_drugs_info_list_and_dict(all_df):
  out_dir = str(pathlib.Path().resolve())
  out_path = f'{out_dir}/datasets/dcdb.csv'

  if 'dcdb.csv' not in os.listdir(f'{out_dir}/datasets/'):
      src_path = f'{out_dir}/datasets/COMPONENTS.txt'
      f = open(src_path, 'r')

      csv_1 = []
      csv_2 = []

      for line in f.readlines():
          each = line.split(' ')[0].split('\t')
          csv_1.append(each[0])
          csv_2.append(each[1])

      # print(len(csv_1))
      # print(len(csv_2))

      csv_1 = np.array(csv_1[1:])
      csv_2 = np.array(csv_2[1:])

      csv_1 = np.reshape(csv_1, (len(csv_1), 1))
      csv_2 = np.reshape(csv_2, (len(csv_2), 1))

      drug_df = pd.DataFrame(np.concatenate([csv_1, csv_2], axis=1), columns=['drug', 'name'])
      drug_df

      drug_df.to_csv(out_path, index=False)
  else:
      drug_df = pd.read_csv(out_path)

  drug_names = dict()

  for drug in drug_df.values:
      drug = list(drug)
      drug_names[drug[0]] = drug[1]

  all_edges = []

  for edge in all_df.values:
      if list(edge) not in all_edges and [edge[1], edge[0]] not in all_edges:
          all_edges.append(list(edge))
          all_edges.append([edge[1], edge[0]])
  return drug_names, all_edges
################################################################
def save_preds_in_csv(all_new_pairs, reverse_node_dict, name):
    d1_names = []
    d2_names = []
    for edge in all_new_pairs:
        a1 = reverse_node_dict[edge[0]]
        a2 = reverse_node_dict[edge[1]]
        d1_names.append(a1)
        d2_names.append(a2)
    all_new_pairs = np.array(all_new_pairs)
    temp_df = pd.DataFrame()
    try:
        temp_df['a1'] = all_new_pairs[:, 0]
        temp_df['a2'] = all_new_pairs[:, 1]
        temp_df['d1'] = d1_names
        temp_df['d2'] = d2_names
    except Exception as e:
        print(name)

    out_dir = str(pathlib.Path().resolve())
    out_path = f'{out_dir}/predictions/{name}.csv'

    if 'predictions' not in os.listdir(out_dir):
        os.mkdir(f'{out_dir}/predictions/')

    temp_df.to_csv(out_path, index=False)
################################################################
def predict_all_top_edges(all_prob_adj_list, data_list):
  all_pairs = []
  for i in range(len(all_prob_adj_list)): # on data_list (features)
    for j in range(len(all_prob_adj_list[i])): # on models
      all_temp_pairs = []
      for k in range(len(all_prob_adj_list[i][j])): # on 5 folds
        prob_adj = all_prob_adj_list[i][j][k]
        high, pairs = predict_top_edges(data_list[i], prob_adj)
        all_temp_pairs.append(pairs)
      all_pairs.append(all_temp_pairs)
      
  return all_pairs
################################################################
def predict_top_new_edges(all_top_pairs):
  all_new_pairs = []
  for pairs in all_top_pairs:
    temp_new_pairs = []
    temp = pairs
    temp_edges = temp[0]
    count = 0
    for edge in temp_edges:
      t = 0
      for i in range(len(temp)):
        e_list = temp[i]
        if edge in e_list:
          t += 1
      if t == len(temp):
        count += 1
        temp_new_pairs.append(edge)
    all_new_pairs.append(temp_new_pairs)
  return all_new_pairs

def predict_all_features_top_edges(all_prob_adj_list, data):
    all_pairs = []
    for j in range(len(all_prob_adj_list)): # on models
        all_temp_pairs = []
        for k in range(len(all_prob_adj_list[j])): # on 5 folds
            prob_adj = all_prob_adj_list[j][k]
            high, pairs = predict_top_edges(data, prob_adj)
            all_temp_pairs.append(pairs)
        all_pairs.append(all_temp_pairs)   
    return all_pairs

def predict_all_features_top_new_edges(all_top_pairs):
    all_new_pairs = []
    for pairs in all_top_pairs:
        temp_new_pairs = []
        temp = pairs
        temp_edges = temp[0]
        count = 0
        for edge in temp_edges:
            t = 0
            for i in range(len(temp)):
                e_list = temp[i]
                if edge in e_list:
                    t += 1
            if t == len(temp):
                count += 1
                temp_new_pairs.append(edge)
        all_new_pairs.append(temp_new_pairs)
    return all_new_pairs