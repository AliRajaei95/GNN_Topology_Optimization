"""
Section IV-B: Prediction-Accuracy case study for ML-assisted topology control.

This script reproduces the prediction-accuracy experiments presented in
Section IV-B of the paper. The case study evaluates and compares multiple
machine-learning architectures for predicting congestion-management and
optimal substation switching (OSS) actions in power systems.

The evaluated ML models include:
    - FNN
    - HomoGCN
    - HomoGAT
    - HomoMPNN
    - HomoEGAT
    - HetEGAT
    - HetMPNN

The experiments are performed on IEEE test systems using datasets generated
from DC optimal power flow (DC-OPF) and DC optimal substation switching
(DC-OSS) simulations under varying operating conditions.

The script supports both classification or Regressio ntasks for topology-control prediction

"""

Folder=""
DataFolder = ""

exec(open(Folder+'DC_Optimal_Splitting_Functions_Pyomo.py').read())
exec(open(Folder+'ML_utilities.py').read())
exec(open(Folder+'FNN_Functions.py').read())
exec(open(Folder+'HomoMPGNN_Functions.py').read())
exec(open(Folder+'HeteroGNN_Functions.py').read())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# %%
# # Run here

# %%


system='118'
dataname='dataset3_gen' 
task = 'Clf'
Num_seeds=1
traintype = 'N0' 

train_datasets = []
for seed in range(0,10):
    File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_N-0_[1]_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'

    with open(File_name, 'rb') as f:
        dataset = pickle.load(f)

    train_datasets.append(dataset)


gamma=2
hops=5
layer=5
hid=64
num_epochs=100

hyper_param = {'gamma':gamma,
               'batch_size':16,
               'split_ratios' : [0.7,0.15,0.15],
               'n_mp_layers':layer,
               'hidden_size':hid,
               'ForceSplit':True,
                'HopsFilter':hops,
                'lr':5e-4,
                'weight_decay':5e-5,
                'Clf_threshold':0.0,
                'clip_val':-0.2,
                'ds_loading':0,
                'by_sample':False
               }






# %%
# FNN
modeltype = 'FNN'

for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'



    clf_res = run_casestudy_FNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


# %%
modeltype = 'HetEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


modeltype = 'HetMPNN'
for seed in range(0,Num_seeds):
    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)

# # %%

modeltype = 'HomoEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


modeltype = 'HomoGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


modeltype = 'HomoMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


modeltype = 'HomoGCN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs2_acc_{modeltype}_{task}_{system}bus_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_N0_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


