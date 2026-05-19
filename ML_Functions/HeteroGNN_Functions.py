"""
Heterogeneous graph neural network (HeteroGNN) utilities for topology-control
prediction in power systems.

This module contains heterogeneous graph neural network architectures,
graph-construction utilities, training pipelines, evaluation routines,
and inference workflows for machine-learning-assisted optimal substation
switching (OSS) and congestion management.

The implementations include:
    - Heterogeneous graph construction using PyTorch Geometric
    - Bus-switchable and non-switchable node representations
    - Edge-aware graph attention networks (EdgeGAT)
    - Heterogeneous message-passing neural networks (HetMPNN)
    - Custom edge-aware attention layers
    - Graph data preprocessing and dataloader utilities
    - Regression and classification training pipelines
    - Boundary-aware regression losses
    - Classification metrics and evaluation utilities
    - Transfer-learning workflows
    - Dataset batching and train/validation/test splitting
    - Model testing and inference routines
    - Visualization of training and validation performance

The HeteroGNN models are designed to predict congestion mitigation and
busbar-splitting actions using heterogeneous graph representations of
power-system operating conditions and network topology.

The framework explicitly distinguishes between switchable and
non-switchable substations, allowing message-passing operations
to better capture topology-control behavior and graph structure.

Developed for machine-learning-assisted power-system operation,
congestion management, and topology-control research using
PyTorch, PyTorch Geometric, CUDA, and Gurobi optimization frameworks.

Tested with Python 3.12, PyTorch 2.6, CUDA 12.4, and Gurobi 10.0.
"""


# Libraries
import torch
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import HeteroConv, GATConv, TransformerConv, MessagePassing
from torch_geometric.loader import DataLoader
import torch_scatter
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



# %%
def create_grid_hetero_data(
    X_bus_feature,           # (sample, bus, Nfeature)
    Y_target_list,                # list of (sw, 1)
    bus_sw_indices_list,       # list of lists or arrays, one per sample
    X_line_feature_list,     #  list (lines, Nf)
    Edge_sender_list,        # list(sample)
    Edge_receiver_list,      # list(sample)
    batch_idx=0              # Batch index to extract (or None for all batches)
):
    """
        Create a heterogeneous graph representation for a single power-system sample.

        This function converts power-system operating-point features into a
        PyTorch Geometric `HeteroData` object for heterogeneous graph neural
        network (HeteroGNN) models.

        The graph explicitly separates:
            - Switchable buses (`bus_sw`)
            - Non-switchable buses (`bus_nosw`)

        The function assigns:
            - Node features for each bus type
            - Target labels for switchable buses
            - Edge indices for all heterogeneous edge types
            - Edge features for transmission lines

        Both forward and reverse transmission-line edges are created to enable
        bidirectional message passing. Reverse-edge power-flow-related features
        are sign-adjusted to preserve physical directionality.

        The heterogeneous graph representation enables the model to distinguish
        between candidate topology-control substations and regular buses during
        message-passing operations.

        Parameters
        ----------
        X_bus_feature : numpy.ndarray
            Bus feature tensor with shape:
                [Nsamples, Nbus, Nbus_features]
        Y_target_list : list
            List of target arrays for switchable buses.
            Each element typically has shape:
                [Nswitchable_bus, 1]
        bus_sw_indices_list : list
            List containing indices of switchable buses for each sample.
        X_line_feature_list : list
            List of line-feature arrays with shape:
                [Nlines, Nline_features]
        Edge_sender_list : list
            Sender-node indices for graph edges.
        Edge_receiver_list : list
            Receiver-node indices for graph edges.
        batch_idx : int
            Sample index used to construct a single heterogeneous graph.

        Returns
        -------
        torch_geometric.data.HeteroData
            Heterogeneous graph object containing:
                - bus_sw node features and labels
                - bus_nosw node features
                - heterogeneous edge indices
                - edge attributes/features
"""
    
    # Create a new HeteroData instance
    data = HeteroData()
    nbus = X_bus_feature.shape[1]

    all_bus_indices = np.arange(nbus)
    bus_sw_indices = bus_sw_indices_list[batch_idx]
    bus_nosw_indices = np.array([b for b in all_bus_indices if b not in bus_sw_indices])
    
    bus_sw_indices_set = set(bus_sw_indices )
    bus_nosw_indices_set = set(all_bus_indices.tolist()) - bus_sw_indices_set

    # Mapping from global index to local index in subgraph
    bus_sw_map = {global_idx: local_idx for local_idx, global_idx in enumerate(bus_sw_indices)}
    bus_nosw_map = {global_idx: local_idx for local_idx, global_idx in enumerate(bus_nosw_indices)}

    
    # Extract features for the specified batch
    if batch_idx is None: 
        raise NotImplementedError("Processing all batches at once is not implemented in this example")
                                                                              #we pass the entire dataset, that's why we need idx 
            
    # Add node features
    data['bus_sw'].x = torch.tensor(X_bus_feature[batch_idx, bus_sw_indices, :], dtype=torch.float)
    data['bus_nosw'].x = torch.tensor(X_bus_feature[batch_idx, bus_nosw_indices, :], dtype=torch.float)


    # Add solution values as target values (y)
    data['bus_sw'].y = torch.tensor(Y_target_list[batch_idx], dtype=torch.float) 

    
    # Edge routing
    senders = Edge_sender_list[batch_idx]
    receivers = Edge_receiver_list[batch_idx]
    features = X_line_feature_list[batch_idx]

    # Containers for edges and attributes
    edge_dict = {
        ('bus_sw', 'line', 'bus_sw'): [],
        ('bus_sw', 'line', 'bus_nosw'): [],
        ('bus_nosw', 'line', 'bus_sw'): [],
        ('bus_nosw', 'line', 'bus_nosw'): []
    }
    attr_dict = {key: [] for key in edge_dict}

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        # Identify types
        if s in bus_sw_map:
            s_type = 'bus_sw'
            s_local = bus_sw_map[s]
        else:
            s_type = 'bus_nosw'
            s_local = bus_nosw_map[s]

        if r in bus_sw_map:
            r_type = 'bus_sw'
            r_local = bus_sw_map[r]
        else:
            r_type = 'bus_nosw'
            r_local = bus_nosw_map[r]

        # Forward edge
        edge_type = (s_type, 'line', r_type)
        edge_dict[edge_type].append([s_local, r_local])
        attr_dict[edge_type].append(features[idx])

        # Reverse edge
        edge_type_rev = (r_type, 'line', s_type)
        edge_dict[edge_type_rev].append([r_local, s_local])
        rev_feat = features[idx].copy()
        rev_feat[0] *= -1  # P_ji = -P_ij
        rev_feat[1] *= -1  # ratio_ji = -ratio_ij
        attr_dict[edge_type_rev].append(rev_feat)

    for edge_type in edge_dict:
        if edge_dict[edge_type]:
            edge_index = torch.tensor(edge_dict[edge_type], dtype=torch.long).T
            edge_attr = torch.tensor(attr_dict[edge_type], dtype=torch.float)
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_attr = edge_attr
    
    return data



def create_dataloader_hetero(
     X_bus_feature,                 
    Y_target_list,
    bus_sw_indices_list,             
    X_line_feature_list,                
    Edge_sender_list,                 
    Edge_receiver_list, 
    batch_size=128,
    shuffle = True
):
    """
    Create a dataloader for the heterogeneous graph data with solutions.
    """
    
    
    # Create a list of HeteroData objects
    dataset = []
    
    for i in range(len(X_bus_feature)):  # Process up to 1000 samples for this example
        data = create_grid_hetero_data(
            X_bus_feature,                  
            Y_target_list,    
            bus_sw_indices_list,        
            X_line_feature_list,                 
            Edge_sender_list,                
            Edge_receiver_list,
            batch_idx=i
        )
        dataset.append(data)
    
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader



# %%

class MLP(torch.nn.Module): 
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

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        # self.x = x  # Save for use in message()
        self.x_src = x_src
        self.x_dst = x_dst
        self.edge_attr = edge_attr  # Save for use in message()
        # Run message passing
        out = self.propagate(edge_index=edge_index,
                             x=x_dst,
                            size=(x_src.size(0), x_dst.size(0)))  # (num_src_nodes, num_dst_nodes)

        # The edge_attr was updated during message() and stored
        return out , self.updated_edge_attr  # (node_embeddings, edge_embeddings)
    

    def message(self, edge_index_i, edge_index_j): #defines how messages are created on each edge.
        x_i = self.x_dst[edge_index_i]
        x_j = self.x_src[edge_index_j]
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
        self.updated_edge_attr = e_ij + edge_msg  # or another residual rule

        return weighted_msg

    def update(self, aggr_out, x): #defines how the node updates its embedding using the aggregated message.
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)+x  # Residual connection 
    

# %% new implementation based on Homo_edgeaware_GAT
class HeteroEdgeGATLayer_New(nn.Module):
    def __init__(self, node_types, edge_types, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_types = node_types
        self.edge_types = edge_types

        # One TransformerConv per edge type
        self.convs = nn.ModuleDict()
        self.node_mlps = nn.ModuleDict()

        for (src, rel, dst) in edge_types:
            key = f"{src}__{rel}__{dst}"
            self.convs[key] = EdgeAwareAttentionLayer(hidden_size)        

        for node_type in node_types:
            self.node_mlps[node_type] = MLP(2 * hidden_size, hidden_size, hidden_size, layers=2)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Store updates
        updated_x_dict = {k: torch.zeros_like(x) for k, x in x_dict.items()}
        updated_edge_attr_dict = {}

        # Process each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            key = f"{src}__{rel}__{dst}"


            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict[edge_type]
            x_src = x_dict[src]
            x_dst = x_dict[dst]

            # Node update
            x_updated, edge_attr_updated = self.convs[key](x_src, x_dst, edge_index, edge_attr)
            updated_edge_attr_dict[edge_type] = edge_attr_updated

            updated_x_dict[dst] += x_updated

            

        # Aggregate node updates per type
        final_x_dict = {}
        for node_type, x in x_dict.items():
            x_updated = updated_x_dict[node_type]
            x_cat = torch.cat([x, x_updated], dim=-1)
            final_x_dict[node_type] = self.node_mlps[node_type](x_cat) + x

        return final_x_dict, updated_edge_attr_dict



# %%
class HetMPNN(MessagePassing):
    def __init__(self, hidden_size):
        super().__init__(aggr='add')
        self.message_mlp = MLP(3 * hidden_size, hidden_size, hidden_size,2)
        self.update_mlp = MLP(2 * hidden_size, hidden_size, hidden_size,2)

    def forward(self, x, edge_index, edge_attr):
        self.x = x  # Save for use in message()
        self.edge_attr = edge_attr  # Save for use in message()

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        return out, self.updated_edge_attr

    def message(self, x_i, x_j, edge_attr):
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        messages = self.message_mlp(message_input)

        # Update edge_attr as well (residual connection)
        self.updated_edge_attr = edge_attr + self.edge_mlp(message_input)

        return messages

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1)) + x 



class HeteroMPNNLayer_New(nn.Module):
    def __init__(self, node_types, edge_types, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_types = node_types
        self.edge_types = edge_types

        # One TransformerConv per edge type
        self.convs = nn.ModuleDict()
        self.edge_updaters = nn.ModuleDict()
        self.node_updaters = nn.ModuleDict()

        

        for (src, rel, dst) in edge_types:
            key = f"{src}__{rel}__{dst}"
            self.convs[key] = HetMPNN(hidden_size)
            self.edge_updaters[key] = MLP(3 * hidden_size, hidden_size, hidden_size, layers=2)


        for node_type in node_types:
            self.node_updaters[node_type] = MLP(2 * hidden_size, hidden_size, hidden_size, layers=2)

    def forward(self, x_dict,edge_indices_dict, edge_features_dict): #edge_index_dict, edge_attr_dict):
        
        updated_edge_features = {}
        
        # Prepare aggregated messages storage
        aggregated_messages = {node_type: torch.zeros_like(feat) 
                              for node_type, feat in x_dict.items()}
        
        # Process each edge type in parallel
        for edge_type, edge_index in edge_indices_dict.items():
            src_type, rel_type, dst_type = edge_type
            edge_key = f"{src_type}__{rel_type}__{dst_type}"
            
            # Get node features for this edge
            src, dst = edge_index
            x_i = x_dict[dst_type][dst]  # Destination nodes
            x_j = x_dict[src_type][src]  # Source nodes
            
            # Update edge features based on edge type
            # if edge_type in self.physical_edge_types:
            edge_feature = edge_features_dict[edge_type]
            edge_msg = torch.cat((x_i, x_j, edge_feature), dim=-1)
            updated_edge = self.edge_updaters[edge_key](edge_msg)
            updated_edge_features[edge_type] = edge_feature + updated_edge  # Residual connection
            
            # torch_scatter is more efficient for message passing
            aggregated_messages[dst_type] = torch_scatter.scatter_add(updated_edge, dst, dim=0, out=aggregated_messages[dst_type])
            aggregated_messages[src_type] = torch_scatter.scatter_add(updated_edge, src, dim=0, out=aggregated_messages[src_type])
        
        
        # Update node features
        updated_nodes = {}
        for node_type, x in x_dict.items():
            # Combine node features with aggregated messages
            node_input = torch.cat((x, aggregated_messages[node_type]), dim=-1)
            node_update = self.node_updaters[node_type](node_input)
            updated_nodes[node_type] = x + node_update  # Residual connection
        
        return updated_nodes, updated_edge_features




# %%

class Hetero_GNN_Network_New(torch.nn.Module):
    def __init__(
        self,
        hidden_size=32,
        n_mp_layers=2,
        n_bus_features=1,
        n_line_features=3,
        output_dim=1,
        HetGNN_type='HetEGAT2'
    ):
        super().__init__()
        
        # Define node and edge types
        self.hidden_size=hidden_size
        self.node_types = ['bus_nosw','bus_sw']
        self.edge_types = [
            ('bus_sw', 'line', 'bus_sw'),
            ('bus_sw', 'line', 'bus_nosw'),
            ('bus_nosw', 'line', 'bus_sw'),
            ('bus_nosw', 'line', 'bus_nosw')
        ]
      
        # Node encoders - separate MLP for each node type
        self.node_encoders = nn.ModuleDict({
            'bus_sw': MLP(n_bus_features, hidden_size, hidden_size, 2),
            'bus_nosw': MLP(n_bus_features, hidden_size, hidden_size, 2)
        })
        
        # Edge encoders - separate MLP for each edge type
        self.edge_encoders = nn.ModuleDict({
            'line': MLP(n_line_features, hidden_size, hidden_size, 2)
        })
        #TODO?

        if HetGNN_type=='HetEGAT2':
            self.layers = nn.ModuleList([
                HeteroEdgeGATLayer_New(self.node_types, self.edge_types, hidden_size)
                for _ in range(n_mp_layers)
            ])
        elif HetGNN_type=='HetMPNN2':
            self.layers = nn.ModuleList([
                HeteroMPNNLayer_New(self.node_types, self.edge_types, hidden_size)
                for _ in range(n_mp_layers)
            ])
        
        # Decoder (input dim includes node + edge context)
        self.node_decoders = nn.ModuleDict({
            'bus_sw': MLP(hidden_size, hidden_size, output_dim, 2, layernorm=False)
        })



    def forward(self, data): #data is a hetrogeneous graph data
        # Encode node features
        
        x_dict = {
            node_type: self.node_encoders[node_type](data[node_type].x)
            for node_type in self.node_types  #if hasattr(data[node_type], 'x') [Batch*Bus,n_bus_features]
        }
                
        # Encode edge features
        edge_feature_dict = {}
        for src, rel, dst in self.edge_types:
            edge_type = (src, rel, dst)
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_feature_dict[edge_type] = self.edge_encoders['line'](data[edge_type].edge_attr)
            else:
            # shape: (num_edges, feature_dim)
                edge_feature_dict[edge_type] = torch.empty(
                (data[edge_type].num_edges if hasattr(data[edge_type], 'num_edges') else 0,
                self.hidden_size),
                device=list(x_dict.values())[0].device
            )
                
        
        edge_index_dict = {}
        for src, rel, dst in self.edge_types:
            edge_type = (src, rel, dst)
            # edge_index_dict[edge_type] = data[edge_type].edge_index
            if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
                edge_index_dict[edge_type] = data[edge_type].edge_index
            else:
                # If no edges of this type exist, create an empty tensor
                edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=x_dict[src].device)

        
        
    
        # Apply GNN layers
        for layer in self.layers:
            x_dict, edge_feature_dict = layer(x_dict, edge_index_dict, edge_feature_dict)
        
        # # Decode node features to get predictions
        output = {}
        output['bus_sw'] = self.node_decoders['bus_sw'](x_dict['bus_sw'])

        
        return output





# %%
def train_model_HeteroGNN(model, trainloader, optimizer,task='Reg',pos_weight=None):
    
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
        pred_dict = model(batch) #pred_dict['bus'] is [batch*Bus,1] , we use batch['bus'].predict_mask to index the bus_sw nodes which gives [batch*sw,1]

        loss = criterion(pred_dict['bus_sw'], batch['bus_sw'].y)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(trainloader)


def validate_model_HeteroGNN(model, val_loader,task='Reg'):
    
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
        # Forward pass
        pred_dict = model(batch) #pred_dict['bus'] is [batch*Bus,1] , we use batch['bus'].predict_mask to index the bus_sw nodes
        loss = criterion(pred_dict['bus_sw'], batch['bus_sw'].y)
        total_loss += loss.item()
        
    return total_loss / len(val_loader)

# %% [markdown]
# # test

# %%
@torch.no_grad()
def test_model_HeteroGNN(model, testloader,task='Reg',Clf_threshold=0.0,by_sample=True):

    model.eval()
    if task=='Reg':
        total_loss = 0
        criterion = nn.MSELoss()
        all_preds = []
        all_targets = []
        all_pred_by_sample = []
        all_target_by_sample = []

        with torch.no_grad():
            for batch in testloader:

                batch = batch.to(device)
        
                pred_dict = model(batch) #pred_dict['bus'] is [batch*Bus,1] , we use batch['bus'].predict_mask to index the bus_sw nodes

                y_pred = pred_dict['bus_sw'].squeeze()  # [N]
                y_target = batch['bus_sw'].y.squeeze()  # [N]
                node_batch = batch['bus_sw'].batch          # [N], tells which node/bus_sw belongs to which batch
                loss = criterion(y_pred, y_target)
                total_loss += loss.item()
                all_preds.append(y_pred.cpu())
                all_targets.append(y_target.cpu())

                #sample-seperation
                if by_sample==True:
                    preds_by_sample = []
                    targets_by_sample = []

                    num_graphs = int(node_batch.max().item()) + 1 #first, we create empty lists that match the number of samples
                    for _ in range(num_graphs):
                        preds_by_sample.append([])
                        targets_by_sample.append([])

                    for i, sample_idx in enumerate(node_batch.tolist()):
                        preds_by_sample[sample_idx].append( y_pred[i].item() )
                        targets_by_sample[sample_idx].append(y_target[i].item())
                    
                    # Accumulate all predictions
                    for i in range(num_graphs):
                        all_pred_by_sample.append(preds_by_sample[i])
                        all_target_by_sample.append(targets_by_sample[i])

                # done


        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if by_sample==True:
            return {'loss':total_loss/len(testloader),
                    'pred':all_preds.detach().cpu().numpy(),
                    'target':all_targets.detach().cpu().numpy(),
                    'pred_by_sample':all_pred_by_sample,
                    'target_by_sample':all_target_by_sample
                    }
        else:
            return {'loss':total_loss/len(testloader),
                    'pred':all_preds.detach().cpu().numpy(),
                    'target':all_targets.detach().cpu().numpy()
                    }
    
    # =========================================================================================
    elif task=='Clf':
        total_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        all_preds = []
        all_targets = []
        all_probs = []
        all_pred_by_sample = []
        all_target_by_sample = []
        all_probs_by_sample = []

        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(device)
        
                pred_dict = model(batch) #pred_dict['bus'] is [batch*Bus,1] , we use batch['bus'].predict_mask to index the bus_sw nodes

                y_pred = pred_dict['bus_sw'].squeeze() #[N]
                y_target = batch['bus_sw'].y.squeeze()  # [N]
                node_batch = batch['bus_sw'].batch     

                total_loss += criterion(y_pred, y_target).item()

                probs = torch.sigmoid(y_pred)
                y_pred = (probs > 0.5).float() #we pass through a sigmoid and put the thereshold at 0.5
                all_preds.append(y_pred.cpu())
                all_targets.append(y_target.cpu())
                all_probs.append(probs.cpu())


                #sample-seperation
                if by_sample==True:
                    preds_by_sample = []
                    targets_by_sample = []
                    probs_by_sample = []

                    num_graphs = int(node_batch.max().item()) + 1 #first, we create empty lists that match the number of samples
                    for _ in range(num_graphs):
                        preds_by_sample.append([])
                        targets_by_sample.append([])
                        probs_by_sample.append([])

                    for i, sample_idx in enumerate(node_batch.tolist()):
                        preds_by_sample[sample_idx].append( y_pred[i].item() )
                        targets_by_sample[sample_idx].append(y_target[i].item())
                        probs_by_sample[sample_idx].append(probs[i].item())
                    
                    # Accumulate all predictions
                    for i in range(num_graphs):
                        all_pred_by_sample.append(preds_by_sample[i])
                        all_target_by_sample.append(targets_by_sample[i])
                        all_probs_by_sample.append(probs_by_sample[i])

                    # done
                

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

        all_pred_by_sample = []
        all_target_by_sample = []
        all_binary_by_sample = []
        
        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(device)
        
                pred_dict = model(batch) #pred_dict['bus'] is [batch*Bus,1] , we use batch['bus'].predict_mask to index the bus_sw nodes

                y_pred = pred_dict['bus_sw'] 
                y_target = batch['bus_sw'].y 
                binary_preds = (y_pred > Clf_threshold).float() #for regression, the thereshold is 0
                
                all_binary_pred.append(binary_preds.cpu())
                all_preds.append(y_pred.cpu())
                all_targets.append(y_target.cpu())

                #sample-seperation
                preds_by_sample = []
                targets_by_sample = []
                binary_by_sample = []

                num_graphs = int(node_batch.max().item()) + 1 #first, we create empty lists that match the number of samples
                for _ in range(num_graphs):
                    preds_by_sample.append([])
                    targets_by_sample.append([])
                    binary_by_sample.append([])

                for i, sample_idx in enumerate(node_batch.tolist()):
                    preds_by_sample[sample_idx].append( y_pred[i].item() )
                    targets_by_sample[sample_idx].append(y_target[i].item())
                    binary_by_sample[sample_idx].append(binary_preds[i].item())
                
                # Accumulate all predictions
                for i in range(num_graphs):
                    all_pred_by_sample.append(preds_by_sample[i])
                    all_target_by_sample.append(targets_by_sample[i])
                    all_binary_by_sample.append(binary_by_sample[i])
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_binary_pred = torch.cat(all_binary_pred)

        metrics = classification_metrics(all_binary_pred, all_targets)

        if by_sample==True:
            return {
                'metrics': metrics,
                'pred':all_preds.detach().cpu().numpy(),
                'binary-pred':all_binary_pred.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy(),
                'pred_by_sample':all_pred_by_sample,
                'target_by_sample':all_target_by_sample,
                'binary_pred_by_sample':all_binary_by_sample,
                }
        else:
            return {
                'metrics': metrics,
                'pred':all_preds.detach().cpu().numpy(),
                'binary-pred':all_binary_pred.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy()
                }






# %%

# # run case study
def run_casestudy_HeteroGNN(train_dataset_list,
                          test_dataset_list='same',
                          HetGNN_type = 'HetEGAT',
                          train_task='Reg',test_task='Clf',hyper_param={},num_epochs=2,seed=0,
                          save_model_path=None, trained_model_path=None,trained_model=None,
                          trnasfer_learning=False):

  """
        Run a complete HeteroGNN case study for topology-control prediction tasks.

        This function prepares graph-based features from one or multiple datasets,
        constructs heterogeneous graph representations, trains a HeteroGNN model,
        evaluates its performance, and returns the trained model together with
        test results.

        The workflow supports both regression and classification tasks for predicting
        power-system congestion-management and optimal substation switching (OSS)
        actions.

        The function performs:
            - Feature preparation using Prepare_Features_GNN()
            - Heterogeneous graph construction with create_grid_hetero_data()
            - Dataset concatenation and preprocessing
            - Train/validation/test splitting
            - Weighted classification handling for imbalanced datasets
            - HeteroGNN model initialization or pretrained-model loading
            - Optional transfer learning
            - Training with early stopping and learning-rate scheduling
            - Validation and testing
            - Regression and classification evaluation
            - Optional model saving and visualization

        Supported HeteroGNN architectures include:
            - HetEGAT
            - HetEGAT2
            - HetMPNN2

        The heterogeneous graph representation separates:
            - Switchable buses (bus_sw)
            - Non-switchable buses (bus_nosw)

        This separation enables the model to explicitly learn topology-control
        behavior and busbar-splitting decisions through heterogeneous
        message-passing operations.

        Parameters
        ----------
        train_dataset_list : list
            List of training datasets generated from DC-OSS simulations.
        test_dataset_list : list or str
            Test datasets. If 'same', training datasets are reused for testing.
        HetGNN_type : str
            Type of heterogeneous GNN architecture to use.
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

  print('\n\n Hetero GNN model  \n\n')

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
      Y_target_list =  res['Y_reg_list']  #[sample,sw,1]
    elif train_task=='Clf':
      Y_target_list =  res['Y_clf_list']


    # Loop over all samples and generate HeteroData objects
    for i in range(X_bus_feature.shape[0]):
        data = create_grid_hetero_data(
            X_bus_feature,
            Y_target_list,
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
        labels = data['bus_sw'].y.view(-1)  # or whichever node type you use for classification
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
      Y_target_list =  res['Y_reg_list']  #[sample,bus_sw,1]
    elif test_task=='Clf':
      Y_target_list =  res['Y_clf_list']


    # Loop over all samples and generate HeteroData objects
    for i in range(X_bus_feature.shape[0]):
        data = create_grid_hetero_data(
            X_bus_feature,
            Y_target_list,
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
    model = Hetero_GNN_Network_New(n_mp_layers=hyper_param['n_mp_layers'] , hidden_size=hyper_param['hidden_size'],
                          n_bus_features=X_bus_feature.shape[2],n_line_features=X_line_feature_list[0].shape[1],
                            HetGNN_type=HetGNN_type  ).to(device)
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    
    
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
    for param in model.layers.parameters():
        param.requires_grad = False
    # for name, param in model.named_parameters():
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
    train_loss = train_model_HeteroGNN(model, train_loader, optimizer,train_task,pos_weight=pos_weight)
    valid_loss = validate_model_HeteroGNN(model, val_loader,train_task)
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
    plt.title(f'HetGNN Training and Validation loss')
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
    res = test_model_HeteroGNN(model, test_loader,task='Reg',by_sample=hyper_param['by_sample'])
    print('MSE loss on test data is %.3f' % res['loss'])
    
  elif train_task=='Clf' and test_task=='Clf':
    res = test_model_HeteroGNN(model, test_loader,task='Clf',by_sample=hyper_param['by_sample'])
    
    print('Loss and metrics on test data is \n', res['loss'])
    print(res['metrics'])
  
  elif train_task=='Reg' and test_task=='Clf':
    res = test_model_HeteroGNN(model, test_loader,task='Reg-then-Clf',by_sample=hyper_param['by_sample'])
    print('Metrics on test data is')
    print(res['metrics'])


  return {
    'test_res':res,
    'model':model
  }


    



