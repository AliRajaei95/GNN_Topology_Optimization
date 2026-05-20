"""
Homogeneous graph neural network (HomoGNN) utilities for topology-control
prediction in power systems.

This module contains graph neural network (GNN) architectures, graph-construction
utilities, training pipelines, evaluation routines, and inference workflows for
machine-learning-assisted optimal substation switching (OSS) and congestion
management.

The implementations include:
    - Homogeneous graph construction for power-system networks
    - Graph data preparation using PyTorch Geometric
    - GCN, GAT, MPNN, and edge-aware attention architectures
    - Message-passing neural network layers
    - Regression and classification training pipelines
    - Boundary-aware regression losses
    - Classification metrics and evaluation utilities
    - Transfer-learning workflows
    - Dataset splitting and batching utilities
    - Model testing and inference routines
    - Visualization of training and validation performance

The HomoGNN models are designed to predict congestion mitigation and
busbar-splitting actions using graph-structured representations of
power-system operating conditions and network topology.

Developed for machine-learning-assisted power-system operation,
congestion management, and topology-control research using
PyTorch, PyTorch Geometric, CUDA, and Gurobi optimization frameworks.

Tested with Python 3.12, PyTorch 2.6, CUDA 12.4, and Gurobi 10.0.
"""

# %%
# Libraries
#%reset 
import torch
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv,GCNConv,TransformerConv
from torch_geometric.loader import DataLoader
import torch_scatter
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData, Data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
def create_grid_homo_data(
     X_bus_feature,              # shape: (samples, buses, bus_features)
    Y_target,                   # shape: (samples, buses, 1)
    bus_sw_indices_list,       # list of lists or arrays, one per sample
    X_line_feature_list,       # list of [n_edges, edge_features]
    Edge_sender_list,          # list of arrays [n_edges]
    Edge_receiver_list,        # list of arrays [n_edges]
    batch_idx=0
):
    """this function creates homogeneous data because we only have one node type."""


    # Number of buses for this sample
    nbus = X_bus_feature.shape[1]

    # Node features
    x = torch.tensor(X_bus_feature[batch_idx], dtype=torch.float)  # [nbus, num_node_features]

    # Target and mask
    y = torch.full((nbus, 1), float('nan'))  # Initialize with NaNs
    predict_mask = torch.zeros(nbus, dtype=torch.bool)
    predict_mask[bus_sw_indices_list[batch_idx]] = True
    y[predict_mask] = torch.tensor(Y_target[batch_idx], dtype=torch.float)[predict_mask]

    # Edge list
    senders = torch.tensor(Edge_sender_list[batch_idx].flatten(), dtype=torch.long)
    receivers = torch.tensor(Edge_receiver_list[batch_idx].flatten(), dtype=torch.long)
    
    edge_index_ij = torch.stack([senders, receivers], dim=0)
    edge_attr_ij = torch.tensor(X_line_feature_list[batch_idx], dtype=torch.float)

    # line ji
    edge_index_ji = torch.stack([receivers, senders], dim=0)
    edge_attr_ji = edge_attr_ij.clone()
    edge_attr_ji[:, 0] = -edge_attr_ji[:, 0] #P_ji = -P_ij
    edge_attr_ji[:, 1] = -edge_attr_ji[:, 1] #ratio_ji = - ratio_ij

    # Combine edges and features
    edge_index = torch.cat([edge_index_ij, edge_index_ji], dim=1)
    edge_attr = torch.cat([edge_attr_ij, edge_attr_ji], dim=0)

    # Build the homogeneous graph object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, predict_mask=predict_mask)

    return data



# %%

class MLP(torch.nn.Module): #I think this is for encoding features, that's why normalization
    def __init__(self, input_size, hidden_size, output_size, layers=2, layernorm=True):
        super().__init__()
        # Use Sequential instead of ModuleList for faster forward pass
        modules = []
        for i in range(layers):
            modules.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                modules.append(torch.nn.ReLU())
        if layernorm:
            modules.append(torch.nn.LayerNorm(output_size))
        
        self.network = torch.nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        # Sequential is faster than iterating through ModuleList
        return self.network(x)

# %%
class Homo_GCN(nn.Module):
    def __init__(self, n_bus_features, hidden_size=32, n_mp_layers=2, output_dim=1):
        super().__init__()

        # Node feature encoder
        self.encoder_node = MLP(n_bus_features, hidden_size, hidden_size, layers=2)

        # Message passing layers (GCN)
        self.message_layers = nn.ModuleList([
            GCNConv(hidden_size, hidden_size)
            for _ in range(n_mp_layers)
        ])

        # Output decoder
        self.decoder_node = MLP(hidden_size, hidden_size, output_dim, layers=2, layernorm=False)

    def forward(self, data):
        x = self.encoder_node(data.x)
        edge_index = data.edge_index

        for conv in self.message_layers:
            x = conv(x, edge_index) + x  # Residual connection

        out = self.decoder_node(x)
        return out





# %% GAT that doesn't use edge attr
class Homo_GATconv(nn.Module):
    def __init__(self, n_bus_features, hidden_size=32, n_mp_layers=2, output_dim=1):
        super().__init__()

        self.encoder_node = MLP(n_bus_features, hidden_size, hidden_size, layers=2)
        # self.encoder_edge = MLP(n_line_features, hidden_size, hidden_size, layers=2)

        self.n_mp_layers = n_mp_layers
        self.message_layers = nn.ModuleList([
            GATConv(hidden_size, hidden_size, heads=1, concat=False)
            for _ in range(n_mp_layers)
        ])

        self.decoder_node = MLP(hidden_size, hidden_size, output_dim, layers=2, layernorm=False)

    def forward(self, data):
        # Node & edge encoders
        x = self.encoder_node(data.x)
        edge_index = data.edge_index
        # edge_attr = self.encoder_edge(data.edge_attr) if hasattr(data, 'edge_attr') else None

        # Message passing
        for layer in self.message_layers:
            x =  layer(x, edge_index) #+ x  # Residual connection

        # Node-level prediction
        out = self.decoder_node(x)
        return out
    

# %% standard MPNN   
class HomoMPNNLayer(MessagePassing):
    def __init__(self, hidden_size):
        super().__init__(aggr='add')
        self.message_mlp = MLP(3 * hidden_size, hidden_size, hidden_size,layers=2)
        self.update_mlp = MLP(2 * hidden_size, hidden_size, hidden_size,layers=2)

    def forward(self, x, edge_index, edge_attr):
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1)) #+ x


class Homo_MPNN_Network(nn.Module):
    def __init__(self, n_bus_features, n_line_features, hidden_size=32, n_mp_layers=2, output_dim=1):
        super().__init__()
        self.encoder_node = MLP(n_bus_features, hidden_size, hidden_size,layers=2)
        self.encoder_edge = MLP(n_line_features, hidden_size, hidden_size,layers=2)

        self.message_layers = nn.ModuleList([
            HomoMPNNLayer(hidden_size) for _ in range(n_mp_layers)
        ])
        self.decoder_node = MLP(hidden_size, hidden_size, output_dim,layers=2,layernorm=False)

    def forward(self, data):
        x = self.encoder_node(data.x)
        edge_attr = self.encoder_edge(data.edge_attr)
        edge_index = data.edge_index

        for layer in self.message_layers:
            x  = layer(x, edge_index, edge_attr)

        return self.decoder_node(x)
    





# %% custom MP-GAT similar to the hetroGNN
class EdgeAwareAttentionLayer(MessagePassing):
    def __init__(self, hidden_size):
        super().__init__(aggr='add')  # or 'mean'
        self.hidden_size = hidden_size

        # Attention mechanism: [x_i || x_j || e_ij]
        self.att_mlp = MLP(3 * hidden_size, hidden_size, 1, layers=1, layernorm=False)

        # Edge update MLP [x_i || x_j || e_ij]
        self.edge_mlp = MLP(3 * hidden_size, hidden_size, hidden_size, layers=2)

        # Node update MLP [x_i || m_i]
        self.node_mlp = MLP(2 * hidden_size, hidden_size, hidden_size, layers=2)

    def forward(self, x, edge_index, edge_attr):
        self.x = x  # Save for use in message()
        self.edge_attr = edge_attr  # Save for use in message()
        # Run message passing
        out = self.propagate(edge_index=edge_index, x=x)

        # The edge_attr was updated during message() and stored
        return out, self.updated_edge_attr
    

    def message(self, edge_index_i, edge_index_j): #defines how messages are created on each edge.
        x_i = self.x[edge_index_i]
        x_j = self.x[edge_index_j]
        e_ij = self.edge_attr

        # Attention weights from x_i and x_j
        att_input = torch.cat([x_i, x_j, e_ij], dim=-1)
        att_logits = self.att_mlp(att_input).squeeze(-1)
        att_weights = torch_scatter.scatter_softmax(att_logits, edge_index_i)

        # Edge update
        edge_input = torch.cat([x_i, x_j, e_ij], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        weighted_msg = edge_msg * att_weights.unsqueeze(-1)

        # Save updated edge_attr for return in forward()
        self.updated_edge_attr = e_ij + weighted_msg  # or another residual rule

        return weighted_msg

    def update(self, aggr_out, x): #defines how the node updates its embedding using the aggregated message.
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)+x  # Residual connection        
    



    

# %%
class Homo_EdgeGAT_Network(nn.Module):
    def __init__(self, n_bus_features, n_line_features, hidden_size=32, n_mp_layers=2, output_dim=1):
        super().__init__()

        # Input encoders
        self.encoder_node = MLP(n_bus_features, hidden_size, hidden_size, layers=2)
        self.encoder_edge = MLP(n_line_features, hidden_size, hidden_size, layers=2)

        # Message passing layers
        self.layers  = nn.ModuleList([
            EdgeAwareAttentionLayer(hidden_size)
            for _ in range(n_mp_layers)
        ])

        # Output decoder
        self.decoder_node = MLP(hidden_size, hidden_size, output_dim, layers=2, layernorm=False)

    def forward(self, data):
        x = self.encoder_node(data.x)
        edge_attr = self.encoder_edge(data.edge_attr) if hasattr(data, 'edge_attr') else None
        edge_index = data.edge_index
        
        # for layer in self.layers:
        #     x = layer(x, edge_index, edge_attr)

        for layer in self.layers :
            x, edge_attr = layer(x, edge_index, edge_attr)
        

        out = self.decoder_node(x)
        return out
    






# %%
def train_model_HomoGNN(model, trainloader, optimizer,task='Reg',pos_weight=None):
    
    model.train()
    total_loss = 0
    if task=='Reg':
        # criterion = nn.MSELoss()
        criterion = BoundaryAwareRegressionLoss() #tailored design

    elif task == 'Clf':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) #we use this so we don't have to include a sigmoid at the end.


    
#     for batch in tqdm(trainloader, desc="Training"):
    for batch in trainloader:
    
        optimizer.zero_grad()
        batch = batch.to(device)

        # Forward pass
        pred = model(batch)  # shape: [num_nodes, 1]

        # Compute loss only for nodes with prediction target
        loss = criterion(pred[batch.predict_mask], batch.y[batch.predict_mask])

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(trainloader)


def validate_model_HomoGNN(model, val_loader,task='Reg'):
    
    model.eval()
    total_loss = 0
    total_loss = 0
    if task=='Reg':
        # criterion = nn.MSELoss()
        criterion = BoundaryAwareRegressionLoss()
    elif task=='Clf':
        criterion = nn.BCEWithLogitsLoss() # we use this becuz we don't aplly a final sigmoid, this loss first passes through a sigmoid
    
    for batch in val_loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred[batch.predict_mask], batch.y[batch.predict_mask])
        total_loss += loss.item()
        
    return total_loss / len(val_loader)


# %%
@torch.no_grad()
def test_model_HomoGNN(model, testloader,task='Reg',Clf_threshold=0.0,by_sample=False):

    model.eval()
    if task=='Reg':
        total_loss = 0
        criterion = nn.MSELoss()
        all_preds = []
        all_targets = []

        all_pred_by_sample = []
        all_target_by_sample = []
        all_mask_by_sample = []
        all_input_by_sample = []

        with torch.no_grad():
            for batch in testloader:

                batch = batch.to(device)

                pred = model(batch)  # [num_nodes, 1]

                y_pred_masked = pred[batch.predict_mask]
                y_target_masked = batch.y[batch.predict_mask]

                loss = criterion(y_pred_masked, y_target_masked)
                total_loss += loss.item()

                all_preds.append(y_pred_masked.cpu())
                all_targets.append(y_target_masked.cpu())

                # Sample-wise storage
                if by_sample==True:
                    y_pred = pred  #shape(batch*sys,1)

                    num_graphs = batch.num_graphs
                    nodes_per_graph = torch.bincount(batch.batch)

                    for i in range(num_graphs):
                        num_nodes = nodes_per_graph[i]

                        pred_i = y_pred[i*num_nodes : (i+1)*num_nodes,0].cpu().numpy()
                        targ_i = batch.y[i*num_nodes : (i+1)*num_nodes,0].cpu().numpy()
                        mask_i = batch.predict_mask[i*num_nodes : (i+1)*num_nodes].cpu().numpy()
                        input_i = batch.x[i*num_nodes : (i+1)*num_nodes,:].cpu().numpy()

                        all_pred_by_sample.append(pred_i)
                        all_target_by_sample.append(targ_i)
                        all_mask_by_sample.append(mask_i)
                        all_input_by_sample.append(input_i)
                  

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if by_sample==True:
           return {'loss':total_loss/len(testloader),
                'pred':all_preds.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy(),
                'pred_by_sample':all_pred_by_sample,
                'target_by_sample':all_target_by_sample,
                'mask_by_sample':all_mask_by_sample,
                'input_by_sample':all_input_by_sample
                }
        else:
            return {'loss':total_loss/len(testloader),
                    'pred':all_preds.detach().cpu().numpy(),
                    'target':all_targets.detach().cpu().numpy()
                    }
    
    
    elif task=='Clf':
        total_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        all_preds = []
        all_targets = []
        all_probs = []

        all_pred_by_sample = []
        all_target_by_sample = []
        all_probs_by_sample = []
        all_mask_by_sample = []
        all_input_by_sample = []

        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(device)
        
                pred = model(batch)

                y_pred = pred[batch.predict_mask]
                y_target = batch.y[batch.predict_mask]
                total_loss += criterion(y_pred, y_target).item()

                probs = torch.sigmoid(y_pred)
                y_pred = (probs > 0.5).float() #we pass through a sigmoid and put the thereshold at 0.5
                all_preds.append(y_pred.cpu())
                all_targets.append(y_target.cpu())
                all_probs.append(probs.cpu())

                # Sample-wise storage
                if by_sample==True:
                    y_pred = pred  #shape(batch*sys,1)

                    # node_batch = batch.batch
                    num_graphs = batch.num_graphs
                    nodes_per_graph = torch.bincount(batch.batch)

                    for i in range(num_graphs):
                        num_nodes = nodes_per_graph[i]

                        pred_i = y_pred[i*num_nodes : (i+1)*num_nodes,0]
                        targ_i = batch.y[i*num_nodes : (i+1)*num_nodes,0]
                        prob_i = torch.sigmoid(pred_i) 
                        pred_bin_i = (prob_i > 0.5).float()
                        mask_i = batch.predict_mask[i*num_nodes : (i+1)*num_nodes]
                        input_i = batch.x[i*num_nodes : (i+1)*num_nodes,:]

                        all_pred_by_sample.append(pred_bin_i.cpu().numpy())
                        all_target_by_sample.append(targ_i.cpu().numpy())
                        all_mask_by_sample.append(mask_i.cpu().numpy())
                        all_probs_by_sample.append(prob_i.cpu().numpy())
                        all_input_by_sample.append(input_i.cpu().numpy())
                

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs) 
        metrics = classification_metrics(all_preds, all_targets,probs=all_probs)

        if by_sample==True:
           return {
                'loss': total_loss/len(testloader),
                'metrics': metrics,
                'pred':all_preds.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy(),
                'probs':all_probs.detach().cpu().numpy(),
                'pred_by_sample':all_pred_by_sample,
                'target_by_sample':all_target_by_sample,
                'probs_by_sample':all_probs_by_sample,
                'mask_by_sample':all_mask_by_sample,
                'input_by_sample' : all_input_by_sample
                }
        else:
            return {
            'loss': total_loss/len(testloader),
                'metrics': metrics,
                'pred':all_preds.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy(),
                'probs':all_probs.detach().cpu().numpy()
                }




    elif task=='Reg-then-Clf':
        total_loss = 0
        # criterion = nn.BCEWithLogitsLoss() can't be used
        all_preds = []
        all_targets = []
        all_binary_pred = []
        
        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(device)
        
                pred = model(batch)
                y_pred = pred[batch.predict_mask]
                y_target = batch.y[batch.predict_mask]
                binary_preds = (y_pred > Clf_threshold).float() #for regression, the thereshold is 0
                
                all_binary_pred.append(binary_preds.cpu())
                all_preds.append(y_pred.cpu())
                all_targets.append(y_target.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_binary_pred = torch.cat(all_binary_pred)

        metrics = classification_metrics(all_binary_pred, all_targets)

        return {
                'metrics': metrics,
                'pred':all_preds.detach().cpu().numpy(),
                'binary-pred':all_binary_pred.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy()
                }





# %%
def run_casestudy_HomoGNN(train_dataset_list,
                        test_dataset_list='same',
                        train_task='Reg',
                        test_task='Clf',
                        hyper_param={},
                        num_epochs=2,
                        seed=0,
                        HomoGNN_type='MP_GAT',
                        save_model_path=None,
                        trained_model_path=None,
                        trained_model=None,
                        trnasfer_learning=False):

  """
    Run a complete HomoGNN case study for topology-control prediction tasks.

    This function prepares graph-based features from one or multiple datasets,
    constructs homogeneous graph representations, trains a HomoGNN model,
    evaluates its performance, and returns the trained model together with
    test results.

    The workflow supports both regression and classification tasks for predicting
    power-system congestion-management and optimal substation switching (OSS)
    actions.

    The function performs:
        - Feature preparation using Prepare_Features_GNN()
        - Homogeneous graph construction with create_grid_homo_data()
        - Dataset concatenation and preprocessing
        - Train/validation/test splitting
        - Weighted classification handling for imbalanced datasets
        - HomoGNN model initialization or pretrained-model loading
        - Optional transfer learning
        - Training with early stopping and learning-rate scheduling
        - Validation and testing
        - Regression and classification evaluation
        - Optional model saving and visualization

    Supported HomoGNN architectures include:
        - HomoGCN
        - HomoGAT
        - HomoMPNN
        - HomoEGAT (edge-aware graph attention)

    Parameters
    ----------
    train_dataset_list : list
        List of training datasets generated from DC-OSS simulations.
    test_dataset_list : list or str
        Test datasets. If 'same', training datasets are reused for testing.
    train_task : str
        Training objective type. Supported options:
            - 'Reg' : regression
            - 'Clf' : classification
    test_task : str
        Testing/evaluation mode. Supported options:
            - 'Reg'
            - 'Clf'
            - 'Reg-then-Clf'
    hyper_param : dict
        Dictionary containing model architecture, preprocessing,
        and training hyperparameters.
    num_epochs : int
        Number of training epochs.
    seed : int
        Random seed used for dataset splitting.
    HomoGNN_type : str
        Type of homogeneous GNN architecture to use.
    save_model_path : str, optional
        Path for saving the trained model.
    trained_model_path : str, optional
        Path to a pretrained model checkpoint.
    trained_model : torch.nn.Module, optional
        Already initialized/trained model object.
    trnasfer_learning : bool
        If True, freezes message-passing layers and performs transfer learning.

    Returns
    -------
    dict
        Dictionary containing:
            - 'test_res' : test predictions and evaluation metrics
            - 'model' : trained PyTorch model
    """
    
  print('\n\n HomoGNN model  \n\n')
  batch_size = hyper_param['batch_size']

  if test_dataset_list == 'same':
    test_dataset_list = deepcopy(train_dataset_list)


  ## train_dataset ===================================================================================================================================
  dataset = []
  for train_dataset in train_dataset_list:

    res = Prepare_Features_GNN(train_dataset,gamma=hyper_param['gamma'],ds_loading=0.0,Nsamples='all',
                                   ForceSplit=hyper_param['ForceSplit'],HopsFilter=hyper_param['HopsFilter'],Clf_threshold=hyper_param['Clf_threshold'])

    X_bus_feature, bus_sw_indices_list, X_line_feature_list, Edge_sender_list, Edge_receiver_list =\
            res['X_bus_feature'], res['bus_sw_indices_list'], res['X_line_feature_list'], res['Edge_sender_list'], res['Edge_receiver_list']
    
    if train_task=='Reg':
      Y_target =  res['Y_reg_homo']  #[sample,all_bus,1]
    elif train_task=='Clf':
      Y_target =  res['Y_clf_homo']


    # Loop over all samples and generate HeteroData objects
    for i in range(X_bus_feature.shape[0]):
        data = create_grid_homo_data(
            X_bus_feature,
            Y_target,
            bus_sw_indices_list,
            X_line_feature_list,
            Edge_sender_list,
            Edge_receiver_list,
            batch_idx=i
        )
        dataset.append(data)

  print(f"Total samples: {len(dataset)}")

  # === Split into train/val/test ===
  train_indices, val_indices, test_indices = train_val_test_split(num_samples=len(dataset),split_ratios=hyper_param['split_ratios'],seed=seed)

  train_dataset = [dataset[i] for i in train_indices]
  val_dataset = [dataset[i] for i in val_indices]
  

  pos_weight = None
  if train_task=='Clf':
    all_labels = []

    for data in train_dataset:
        # Flatten and collect all labels from this graph
        labels = data.y[data.predict_mask].view(-1)  # or whichever node type you use for classification
        all_labels.append(labels)

    # 2. Concatenate into a single tensor
    all_labels = torch.cat(all_labels, dim=0)

    # 3. Count positives and negatives
    num_pos = (all_labels == 1).sum().item()
    num_neg = (all_labels == 0).sum().item()

    pos_weight = torch.tensor([num_neg / (num_pos + 1e-6)])  # avoid division by zero

#   pos_weight = pos_weight.to(device)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


  ## test_dataset ===================================================================================================================================
  dataset = []
  for test_dataset in test_dataset_list:

    res = Prepare_Features_GNN(test_dataset,gamma=hyper_param['gamma'],ds_loading=0.0,Nsamples='all',
                                   ForceSplit=hyper_param['ForceSplit'],HopsFilter=hyper_param['HopsFilter'],Clf_threshold=hyper_param['Clf_threshold'])

    X_bus_feature, bus_sw_indices_list, X_line_feature_list, Edge_sender_list, Edge_receiver_list =\
            res['X_bus_feature'], res['bus_sw_indices_list'], res['X_line_feature_list'], res['Edge_sender_list'], res['Edge_receiver_list']
    
    if test_task=='Reg':
      Y_target =  res['Y_reg_homo']  #[sample,all_bus,1]
    elif test_task=='Clf':
      Y_target =  res['Y_clf_homo']


    # Loop over all samples and generate HeteroData objects
    for i in range(X_bus_feature.shape[0]):
        data = create_grid_homo_data(
            X_bus_feature,
            Y_target,
            bus_sw_indices_list,
            X_line_feature_list,
            Edge_sender_list,
            Edge_receiver_list,
            batch_idx=i
        )
        dataset.append(data)

  print(f"Total samples: {len(dataset)}")
  

  # === Split into train/val/test ===
  train_indices, val_indices, test_indices = train_val_test_split(num_samples=len(dataset),split_ratios=hyper_param['split_ratios'],seed=seed)
  test_dataset = [dataset[i] for i in test_indices]
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


  # ==================================================================================================================================================================


  #define model
  if trained_model is not None:
    model = trained_model.to(device)
    print("Using provided trained model in memory.")
  else:
    if HomoGNN_type == 'HomoEGAT':
        model = Homo_EdgeGAT_Network(n_mp_layers=hyper_param['n_mp_layers'] , hidden_size=hyper_param['hidden_size'],
                          n_bus_features=X_bus_feature.shape[2],n_line_features=X_line_feature_list[0].shape[1]  ).to(device)
    elif HomoGNN_type == 'HomoGAT':
       model = Homo_GATconv(n_mp_layers=hyper_param['n_mp_layers'] , hidden_size=hyper_param['hidden_size'],
                          n_bus_features=X_bus_feature.shape[2]).to(device)
    elif HomoGNN_type == 'HomoMPNN':
       model = Homo_MPNN_Network(n_mp_layers=hyper_param['n_mp_layers'] , hidden_size=hyper_param['hidden_size'],
                          n_bus_features=X_bus_feature.shape[2],n_line_features=X_line_feature_list[0].shape[1]).to(device)
    elif HomoGNN_type == 'HomoGCN':
       model = Homo_GCN(n_mp_layers=hyper_param['n_mp_layers'] , hidden_size=hyper_param['hidden_size'],
                          n_bus_features=X_bus_feature.shape[2]).to(device)
       

  tot_params = 0
  for parameter in model.parameters():
    layer_ws = 1
    for val in parameter.shape:
        layer_ws *= val
    tot_params += layer_ws
  print(f"Total number of parameters = {tot_params}")

  if trained_model_path is not None:
      model.load_state_dict(torch.load(trained_model_path,map_location=device, weights_only=True))
      print(f"Loaded pretrained model from {trained_model_path}")

  
  if trnasfer_learning == True:
    if HomoGNN_type == 'HomoEGAT':
       for param in model.layers.parameters():
            param.requires_grad = False
    else:   
        for param in model.message_layers.parameters():
            param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    print('transfer learning activated!')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyper_param['lr'], weight_decay=hyper_param['weight_decay'])   
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_param['lr'], weight_decay=hyper_param['weight_decay'])
  
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
  training_losses = []
  validation_losses = []
  best_valid_loss = float('inf')
  best_epoch = -1
  patience_counter = 0
  patience_limit = 20  # stop if no improvement after 30 epochs
  if train_task == 'Clf':
    tol = 1e-4
  elif train_task=='Reg':
      tol = 1e-5

  # num_epochs = 2
  for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    train_loss = train_model_HomoGNN(model, train_loader, optimizer,train_task,pos_weight=pos_weight)
    valid_loss = validate_model_HomoGNN(model, val_loader,train_task)
    training_losses.append(train_loss)
    validation_losses.append(valid_loss)

    # wandb.log({"training_loss": train_loss, "validation_loss": valid_loss})

    scheduler.step(train_loss)

    if epoch % 50 == 0:
      print(f'Epoch: {epoch}')
      print(f'\tTrain Loss: {train_loss:.4f}')
      print(f'\t Val. Loss: {valid_loss:.4f}')
    
    # Early stopping check
    if valid_loss < best_valid_loss - tol:  # use a small epsilon to avoid micro fluctuations
        best_valid_loss = valid_loss
        best_epoch = epoch
        patience_counter = 0
        best_model_state = model.state_dict()  # save best model
    else:
        patience_counter += 1

    if patience_counter >= patience_limit:
        print(f"Early stopping at epoch {epoch}. Best validation loss: {best_valid_loss:.4f} at epoch {best_epoch}")
        break

  # Optional: restore best model
  if 'best_model_state' in locals():
    model.load_state_dict(best_model_state)
    print(f"Model restored to best state from epoch {best_epoch}")

  if num_epochs>1:
    plt.subplots(figsize=(5,3))
    plt.plot([i for i in range(len(training_losses))], training_losses, 'r', label='Training loss')
    plt.plot([i for i in range(len(validation_losses))], validation_losses, 'g', label='Validation loss')
    plt.legend()
    plt.title(f'HomoGNN Training and Validation loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    # plt.semilogy()
    plt.show()

  if num_epochs > 0:
    training_losses = np.array(training_losses)
    validation_losses = np.array(validation_losses)

  if save_model_path is not None:
      torch.save(model.state_dict(), save_model_path)
      print(f"Saved trained model to {save_model_path}")

  # testing
  if train_task=='Reg' and test_task=='Reg':
    res = test_model_HomoGNN(model, test_loader,task='Reg',by_sample=hyper_param['by_sample'])
    print('MSE loss on test data is %.3f' % res['loss'])

  elif train_task=='Clf' and test_task=='Clf':
    res = test_model_HomoGNN(model, test_loader,task='Clf',by_sample=hyper_param['by_sample'])
    print('Loss and metrics on test data is \n', res['loss'])
    print(res['metrics'])
  
  elif train_task=='Reg' and test_task=='Clf':
    res = test_model_HomoGNN(model, test_loader,task='Reg-then-Clf')
    print('Metrics on test data is')
    print(res['metrics'])


  return {
    'test_res':res,
    'model':model
  }


