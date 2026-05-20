"""
Section IV-C: Generalization-to-topology-changes case study for ML-assisted
topology control.

This script reproduces the topology-generalization experiments presented
in Section IV-C of the paper. The case study evaluates how well different
machine-learning architectures generalize to unseen transmission-line
outage conditions and changing network topologies.


The experiments analyze several generalization settings, including:
    - Training on N-0 and testing on N-1 contingencies
    - Training on N-1 and testing on N-2 contingencies
    - Training on combined N-0/N-1/N-2 datasets
    - Same-topology vs unseen-topology evaluation

The evaluated ML models include:
    - FNN
    - HomoGCN
    - HomoGAT
    - HomoMPNN
    - HomoEGAT
    - HetEGAT
    - HetMPNN


This case study demonstrates the robustness and transferability of
graph-based machine-learning models for topology-control prediction
under changing power-system network conditions.
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


# %% [markdown]
# # Run here


gamma=2
hops=5
Num_seeds=1
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
traintype='N0'
system='118'
dataname='dataset3_gen' 

train_datasets = []
Nkname = 'N-0_[1]'


for seed in range(0,10):
    if system == '118' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'

    with open(File_name, 'rb') as f:
        dataset = pickle.load(f)

    train_datasets.append(dataset)


# %%
testtype='N1'
test_datasets = []
Nkname = 'N-1_[0, 1]'

for seed in range(0,10):
    if system == '118' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'

    with open(File_name, 'rb') as f:
        dataset = pickle.load(f)
    test_datasets.append(dataset)

# %%
modeltype = 'FNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_FNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_FNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model=clf_res['model'],
                                 train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# %%

# HeteroGNN
modeltype = 'HetEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model=clf_res['model'],
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=1,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# ===========

modeltype = 'HetMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model=clf_res['model'],
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# %%

# HomoGNN
modeltype = 'HomoEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)




#=

# HomoGNN
modeltype = 'HomoGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


#=

# HomoGNN
modeltype = 'HomoMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


#=

# HomoGNN
modeltype = 'HomoGCN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)





# %%


# %% [markdown]
# # N-1 to N-2
traintype='N1'
train_datasets = []

Nkname = 'N-1_[0, 1]'

for seed in range(0,10):
    if system == '14' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/14-bus/{dataname}/Dataset_OSS_14bus_{Nkname}_1000samples_1.5-2.5demand_0.75-0.75limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '14' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/14-bus/{dataname}/Dataset_OSS_14bus_{Nkname}_1000samples_1.5-2.5demand_0.75-0.75limit_0.8-1.2gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '118' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_1-1gen_0.95congfilter_10hops_seed{seed}.pkl'
    if system == '118' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'
    elif system == '300' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_{Nkname}_1000samples_0.6-1demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '300' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_N-0_[1]_1000samples_0.6-1demand_1-1limit_0.8-1.2gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '500' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/500-bus/{dataname}/Dataset_OSS_500bus_N-0_[1]_250samples_0.8-1.2demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'

    with open(File_name, 'rb') as f:
        dataset = pickle.load(f)

    train_datasets.append(dataset)



testtype='N2'
test_datasets = []

Nkname = 'N-2_[0, 0, 1]'

for seed in range(0,10):
    if system == '14' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/14-bus/{dataname}/Dataset_OSS_14bus_{Nkname}_1000samples_1.5-2.5demand_0.75-0.75limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '14' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/14-bus/{dataname}/Dataset_OSS_14bus_{Nkname}_1000samples_1.5-2.5demand_0.75-0.75limit_0.8-1.2gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '118' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_1-1gen_0.95congfilter_10hops_seed{seed}.pkl'
    if system == '118' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'
    elif system == '300' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_{Nkname}_1000samples_0.6-1demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '300' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_N-0_[1]_1000samples_0.6-1demand_1-1limit_0.8-1.2gen_0.95congfilter_5hops_seed{seed}.pkl'
    elif system == '500' and dataname == 'dataset1_1lim':
        File_name = DataFolder + f'Datasets/500-bus/{dataname}/Dataset_OSS_500bus_N-0_[1]_250samples_0.8-1.2demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'


    with open(File_name, 'rb') as f:
        dataset = pickle.load(f)
    test_datasets.append(dataset)


# %%
modeltype = 'FNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_FNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_FNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model=clf_res['model'],
                                 train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)






# %%

# HeteroGNN
modeltype = 'HetEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model=clf_res['model'],
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# ===========

modeltype = 'HetMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model=clf_res['model'],
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# %%

# HomoGNN
modeltype = 'HomoEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)




#=

# HomoGNN
modeltype = 'HomoGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


#=

# HomoGNN
modeltype = 'HomoMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


#=

# HomoGNN
modeltype = 'HomoGCN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # generalization
    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_{testtype}_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model=clf_res['model'],
                               hyper_param=hyper_param,num_epochs=0,seed=seed)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)





# %% [markdown]
# # combine all datasets

# %%
traintype='N0N1N2'
train_datasets = []

for seed in range(0,10):
    for Nkname in [ 'N-0_[1]', 'N-1_[0, 1]' , 'N-2_[0, 0, 1]'  ]:
        if system == '14' and dataname == 'dataset1_1lim':
            File_name = DataFolder + f'Datasets/14-bus/{dataname}/Dataset_OSS_14bus_{Nkname}_1000samples_1.5-2.5demand_0.75-0.75limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
        elif system == '14' and dataname == 'dataset3_gen':
            File_name = DataFolder + f'Datasets/14-bus/{dataname}/Dataset_OSS_14bus_{Nkname}_1000samples_1.5-2.5demand_0.75-0.75limit_0.8-1.2gen_0.95congfilter_5hops_seed{seed}.pkl'
        elif system == '118' and dataname == 'dataset1_1lim':
            File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_1-1gen_0.95congfilter_10hops_seed{seed}.pkl'
        if system == '118' and dataname == 'dataset3_gen':
            File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'
        elif system == '300' and dataname == 'dataset1_1lim':
            File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_{Nkname}_1000samples_0.6-1demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
        elif system == '300' and dataname == 'dataset3_gen':
            File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_N-0_[1]_1000samples_0.6-1demand_1-1limit_0.8-1.2gen_0.95congfilter_5hops_seed{seed}.pkl'
        elif system == '500' and dataname == 'dataset1_1lim':
            File_name = DataFolder + f'Datasets/500-bus/{dataname}/Dataset_OSS_500bus_N-0_[1]_250samples_0.8-1.2demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'

        with open(File_name, 'rb') as f:
            dataset = pickle.load(f)

        train_datasets.append(dataset)





# %%
modeltype = 'FNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_FNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# %%

# HeteroGNN
modeltype = 'HetEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)




# ===========

modeltype = 'HetMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                               test_dataset_list='same',
                               HetGNN_type = modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)




# %%

# HomoGNN
modeltype = 'HomoEGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)






#=

# HomoGNN
modeltype = 'HomoGAT'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    

#=

# HomoGNN
modeltype = 'HomoMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


   


#=

# HomoGNN
modeltype = 'HomoGCN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs3_{system}bus_{task}_{modeltype}_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    save_clf_model = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'
    trained_model_path = None #Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'


    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,
                                save_model_path=save_clf_model,trained_model_path=trained_model_path)
    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    


