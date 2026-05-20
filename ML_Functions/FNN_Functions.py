"""
Feedforward neural network (FNN) utilities for topology-control prediction.

This module contains neural-network architectures, training pipelines,
evaluation utilities, and inference routines for machine-learning-based
optimal substation switching (OSS) and congestion-management studies.

The implementations include:
    - Feedforward neural network (FNN/MLP) models
    - Regression and classification training pipelines
    - Boundary-aware regression losses
    - Classification metrics and evaluation utilities
    - Dataset preparation and train/validation/test splitting
    - Model testing and inference routines
    - Visualization utilities for prediction analysis
    - Power-system topology-control prediction workflows

The models are designed to learn congestion mitigation and busbar splitting
actions from DC-OSS-generated datasets and engineered graph-based features.

Developed for machine-learning-assisted power-system operation,
congestion management, and topology-control research using
PyTorch, PyTorch Geometric, and Gurobi optimization frameworks.

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
import networkx as nx
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm




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
# FNN
def train_model_FNN(model, loader, optimizer,task='Reg',pos_weight=None):
    
    model.train()
    total_loss = 0
    if task=='Reg':
        # criterion = nn.MSELoss()
        criterion = BoundaryAwareRegressionLoss()

    elif task == 'Clf':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) 

    
    for inputs, targets, pred_mask in loader:

        inputs = inputs.to(device)
        targets = targets.to(device)
        pred_mask = pred_mask.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        # Apply mask to prediction and target
        outputs = outputs[pred_mask]
        targets = targets[pred_mask]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(loader)


def validate_model_FNN(model, loader,task='Reg'):
    
    model.eval()
    total_loss = 0
    if task=='Reg':
        # criterion = nn.MSELoss()
        criterion = BoundaryAwareRegressionLoss()
    elif task=='Clf':
        criterion = nn.BCEWithLogitsLoss() # we use this becuz we don't aplly a final sigmoid, this loss first passes through a sigmoid

    with torch.no_grad():
        for inputs, targets, pred_mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred_mask = pred_mask.to(device)
            outputs = model(inputs)
            outputs = outputs[pred_mask]
            targets = targets[pred_mask]
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss/len(loader)



# %% [markdown]
# test ===================================================================================================
@torch.no_grad()
def test_model_FNN(model, loader,task='Reg',Clf_threshold=0.0,by_sample=True):
    
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
            for inputs, targets,pred_mask  in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred_mask = pred_mask.to(device)
                outputs = model(inputs)


                masked_outputs = outputs[pred_mask]
                masked_targets = targets[pred_mask]
                loss = criterion(masked_outputs, masked_targets)
                total_loss += loss.item()

                all_preds.append(masked_outputs.cpu())
                all_targets.append(masked_targets.cpu())

                # Sample-wise storage
                if by_sample==True:
                  for i in range(outputs.size(0)):
                      input_i = inputs[i].cpu().numpy()
                      mask_i = pred_mask[i].cpu().numpy()
                      pred_i = outputs[i].cpu().numpy()
                      targ_i = targets[i].cpu().numpy()
                      all_pred_by_sample.append(pred_i)
                      all_target_by_sample.append(targ_i)
                      all_mask_by_sample.append(mask_i)
                      all_input_by_sample.append(input_i)
                

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if by_sample==True:
           return {'loss':total_loss/len(loader),
                'pred':all_preds.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy(),
                'pred_by_sample':all_pred_by_sample,
                'target_by_sample':all_target_by_sample,
                'mask_by_sample':all_mask_by_sample,
                'input_by_sample':all_input_by_sample
                }
        else:
            return {'loss':total_loss/len(loader),
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
            for inputs, targets, pred_mask in loader:

                inputs = inputs.to(device)
                targets = targets.to(device)
                pred_mask = pred_mask.to(device)

                outputs = model(inputs)

                outputs_masked = outputs[pred_mask]
                targets_masked = targets[pred_mask]

                total_loss += criterion(outputs_masked, targets_masked).item()

                probs_masked = torch.sigmoid(outputs_masked)
                preds_masked = (probs_masked > 0.5).float()

                all_preds.append(preds_masked.cpu())
                all_targets.append(targets_masked.cpu())
                all_probs.append(probs_masked.cpu())

                # --- Sample-wise Storage ---
                if by_sample==True:
                  for i in range(outputs.size(0)):
                      input_i = inputs[i]
                      mask_i = pred_mask[i]
                      pred_i = outputs[i]
                      target_i = targets[i]

                      prob_i = torch.sigmoid(pred_i)
                      pred_bin_i = (prob_i > 0.5).float()

                      all_pred_by_sample.append(pred_bin_i.cpu().numpy())
                      all_target_by_sample.append(target_i.cpu().numpy())
                      all_probs_by_sample.append(prob_i.cpu().numpy())
                      all_mask_by_sample.append(mask_i.cpu().numpy())
                      all_input_by_sample.append(input_i.cpu().numpy())


        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)

        metrics = classification_metrics(all_preds, all_targets,probs=all_probs)

        if by_sample==True:
           return {
            'loss': total_loss/len(loader),
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
            'loss': total_loss/len(loader),
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
        all_binary_pred_by_sample = []

        with torch.no_grad():
            for inputs, targets, pred_mask in loader:

                inputs = inputs.to(device)
                targets = targets.to(device)
                pred_mask = pred_mask.to(device)

                outputs = model(inputs)
                
                outputs_masked = outputs[pred_mask]
                targets_masked = targets[pred_mask]
                binary_preds_masked = (outputs_masked > Clf_threshold).float()

                all_preds.append(outputs_masked.cpu())
                all_targets.append(targets_masked.cpu())
                all_binary_pred.append(binary_preds_masked.cpu())

                # Sample-wise masking and saving
                if by_sample==True:
                  for i in range(outputs.size(0)):
                      mask_i = pred_mask[i]
                      pred_i = outputs[i][mask_i]
                      target_i = targets[i][mask_i]
                      binary_i = (pred_i > Clf_threshold).float()

                      all_pred_by_sample.append(pred_i.cpu().numpy())
                      all_target_by_sample.append(target_i.cpu().numpy())
                      all_binary_pred_by_sample.append(binary_i.cpu().numpy())
        
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
                'binary_by_sample':all_probs_by_sample
                }
        else:
           return {
                'metrics': metrics,
                'pred':all_preds.detach().cpu().numpy(),
                'binary-pred':all_binary_pred.detach().cpu().numpy(),
                'target':all_targets.detach().cpu().numpy()
                }
    



# %%


# %%
def run_casestudy_FNN(train_dataset_list,
                      test_dataset_list='same',
                      train_task='Reg',
                      test_task='Clf',
                      hyper_param={},
                      num_epochs=2,
                      seed=0,
                      save_model_path=None,
                      trained_model_path=None,
                      trained_model=None):

  """
    Run a complete feedforward neural network (FNN) case study for topology-control
    prediction tasks.

    This function prepares input features and targets from one or multiple datasets,
    constructs train/validation/test splits, trains an FNN model, evaluates its
    performance, and returns the trained model together with test results.

    The workflow supports both regression and classification tasks for predicting
    power-system congestion-management and optimal substation switching (OSS)
    actions.

    The function performs:
        - Feature preparation using Prepare_Features_GNN()
        - Dataset concatenation and preprocessing
        - Tensor conversion and masking
        - Train/validation/test splitting
        - Weighted classification handling for imbalanced datasets
        - FNN model initialization or pretrained-model loading
        - Training with early stopping and learning-rate scheduling
        - Validation and testing
        - Regression and classification evaluation
        - Optional model saving and prediction visualization

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
        Dictionary containing model architecture, preprocessing, and training
        hyperparameters.
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

    Returns
    -------
    dict
        Dictionary containing:
            - 'test_res' : test predictions and evaluation metrics
            - 'model' : trained PyTorch model
"""
  
  print('\n\n FNN model  \n\n')

  batch_size = hyper_param['batch_size']

  if test_dataset_list == 'same':
    test_dataset_list = deepcopy(train_dataset_list)

    

  ## train_dataset ===================================================================================================================================
  X_batches = []
  Y_bathes = []
  mask_batches = []
  for train_dataset in train_dataset_list:

    res = Prepare_Features_GNN(train_dataset,gamma=hyper_param['gamma'],ds_loading=hyper_param['ds_loading'],Nsamples='all',ForceSplit=hyper_param['ForceSplit'],
                                    HopsFilter=hyper_param['HopsFilter'],Clf_threshold=hyper_param['Clf_threshold'],clip_val=hyper_param['clip_val'])

    X_bus_feature, bus_sw_indices_list, X_line_feature_FNN, prediction_mask =\
            res['X_bus_feature'], res['bus_sw_indices_list'], res['X_line_feature_FNN'], res['prediction_mask']
  
    if train_task=='Reg':
      Y_target =  res['Y_reg_homo']
    elif train_task=='Clf':
      Y_target =  res['Y_clf_homo']
    
    Nsamples = X_bus_feature.shape[0]

    X_bus_feature = X_bus_feature.reshape((Nsamples,-1))
    X_line_feature_FNN = X_line_feature_FNN.reshape((Nsamples,-1))
    X_features = np.concatenate((X_bus_feature,X_line_feature_FNN),axis=1)

    Y_target = Y_target.reshape((Nsamples,-1))
  
    X_batches.append(X_features)
    Y_bathes.append(Y_target)
    mask_batches.append(prediction_mask)

  # combine all
  X_features = np.vstack(X_batches)
  Y_target = np.vstack(Y_bathes)
  prediction_mask = np.vstack(mask_batches)

  Nsamples = X_features.shape[0]
  print(f"Total samples: {Nsamples}")

  X_tensor = torch.tensor(X_features, dtype=torch.float32)
  Y_tensor = torch.tensor(Y_target, dtype=torch.float32)
  prediction_mask = torch.tensor(prediction_mask, dtype=torch.bool)

  train_indices, val_indices, test_indices = train_val_test_split(num_samples=Nsamples,split_ratios=hyper_param['split_ratios'],seed=seed)

  pos_weight = None
  if train_task=='Clf':
    train_labels = Y_tensor[train_indices][prediction_mask[train_indices]].view(-1)
    num_positives = (train_labels == 1).sum().item()
    num_negatives = (train_labels == 0).sum().item()

    # Avoid division by zero
    if num_positives == 0:
        pos_weight = torch.tensor(1.0, device=device)  # fallback
    else:
        pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32, device=device)

    print(f"pos_weight for BCEWithLogitsLoss: {pos_weight.item():.2f}")



  train_tensordataset = TensorDataset(X_tensor[train_indices], Y_tensor[train_indices],prediction_mask[train_indices])
  val_tensordataset   = TensorDataset(X_tensor[val_indices], Y_tensor[val_indices],prediction_mask[val_indices])

  train_loader = DataLoader(train_tensordataset, batch_size=batch_size, shuffle=True)
  val_loader   = DataLoader(val_tensordataset, batch_size=batch_size, shuffle=False)



  ## test_dataset ===================================================================================================================================

  X_batches = []
  Y_bathes = []
  mask_batches = []

  for test_dataset in test_dataset_list:
    res = Prepare_Features_GNN(test_dataset,gamma=hyper_param['gamma'],ds_loading=hyper_param['ds_loading'],Nsamples='all',ForceSplit=hyper_param['ForceSplit'],
                                   HopsFilter=hyper_param['HopsFilter'],Clf_threshold=hyper_param['Clf_threshold'],clip_val=hyper_param['clip_val'])

    X_bus_feature, bus_sw_indices_list, X_line_feature_FNN, prediction_mask =\
            res['X_bus_feature'], res['bus_sw_indices_list'], res['X_line_feature_FNN'], res['prediction_mask']
    
    if test_task=='Reg':
      Y_target =  res['Y_reg_homo']
    elif test_task=='Clf':
      Y_target =  res['Y_clf_homo']

    Nsamples = X_bus_feature.shape[0]
      
    X_bus_feature = X_bus_feature.reshape((Nsamples,-1))
    X_line_feature_FNN = X_line_feature_FNN.reshape((Nsamples,-1))
    X_features = np.concatenate((X_bus_feature,X_line_feature_FNN),axis=1)
    
    Y_target = Y_target.reshape((Nsamples,-1))

    X_batches.append(X_features)
    Y_bathes.append(Y_target)
    mask_batches.append(prediction_mask)

  # combine all
  X_features = np.vstack(X_batches)
  Y_target = np.vstack(Y_bathes)
  prediction_mask = np.vstack(mask_batches)

  Nsamples = X_features.shape[0]
  print(f"Total samples: {Nsamples}")
    

  X_tensor = torch.tensor(X_features, dtype=torch.float32)
  Y_tensor = torch.tensor(Y_target, dtype=torch.float32)
  prediction_mask = torch.tensor(prediction_mask, dtype=torch.bool)

  train_indices, val_indices, test_indices = train_val_test_split(num_samples=Nsamples, split_ratios=hyper_param['split_ratios'],seed=seed)
  
  test_tensordataset  = TensorDataset(X_tensor[test_indices], Y_tensor[test_indices],prediction_mask[test_indices])

  test_loader  = DataLoader(test_tensordataset, batch_size=batch_size, shuffle=False)


  # ==================================================================================================================================================================

  #define model
  if trained_model is not None:
    model = trained_model.to(device)
    print("Using provided trained model in memory.")
  else:
     model = MLP(input_size=X_tensor.shape[1],hidden_size=hyper_param['hidden_size'],layers=hyper_param['n_mp_layers'],output_size=Y_tensor.shape[1],layernorm=False ).to(device)
  
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
  
  for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    train_loss = train_model_FNN(model, train_loader, optimizer,train_task,pos_weight=pos_weight)
    valid_loss = validate_model_FNN(model, val_loader,train_task)
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
    plt.title(f'FNN Training and Validation loss')
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
    res = test_model_FNN(model, test_loader,task='Reg',by_sample=hyper_param['by_sample'])
    print('MSE loss on test data is %.3f' % res['loss'])
    # plot_prediction_scatter(res['pred'],res['target'])

  elif train_task=='Clf' and test_task=='Clf':
    res = test_model_FNN(model, test_loader,task='Clf',by_sample=hyper_param['by_sample'])
    print('Loss and metrics on test data is \n', res['loss'])
    print(res['metrics'])

  
  elif train_task=='Reg' and test_task=='Clf':
    res = test_model_FNN(model, test_loader,task='Reg-then-Clf',Clf_threshold=hyper_param['Clf_threshold'],by_sample=hyper_param['by_sample'])
    print(res['pred'].shape)
    print('Metrics on test data is')
    print(res['metrics'])

  

  return {
    'test_res':res,
    'model':model
  }

