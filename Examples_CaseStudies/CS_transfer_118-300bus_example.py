"""
Section IV-D: Transferability-across-systems case study for ML-assisted
topology control (transfer learning).

This script reproduces the transfer-learning experiments presented
in Section IV-D of the paper. The case study evaluates how pretrained
machine-learning models can be adapted to new power-system networks
through fine-tuning and transfer learning.

The workflow investigates whether graph neural network models trained on
one power-system network can efficiently transfer learned representations
and topology-control knowledge to another system with limited retraining.

The experiments analyze cross-system transfer scenarios such as:
    - Pretraining on IEEE 118-bus systems
    - Fine-tuning on IEEE 300-bus systems
    - Evaluating pretrained vs transfer-learned performance

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
               'split_ratios' : [0.7,0.1,0.2],
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
traintype='N0N1N2'
train_system='300'
dataname='dataset3_gen'
task = 'Clf'


train_datasets = []

for seed in range(0,1):
    for Nkname in [  'N-0_[1]', 'N-1_[0, 1]' , 'N-2_[0, 0, 1]' ]: 
        if train_system == '118' and dataname == 'dataset3_gen':
            File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'

        with open(File_name, 'rb') as f:
            dataset = pickle.load(f)

        train_datasets.append(dataset)


# %%
testtype='N0N1N2' 
test_system = '300'
test_datasets = []

for seed in range(0,10):
    for Nkname in [ 'N-0_[1]', 'N-1_[0, 1]' , 'N-2_[0, 0, 1]' ]: 
        if test_system == '300' and dataname == 'dataset1_1lim':
            File_name = DataFolder + f'Datasets/300-bus/{dataname}/Dataset_OSS_300bus_{Nkname}_1000samples_0.6-1demand_1-1limit_1-1gen_0.95congfilter_5hops_seed{seed}.pkl'
        
        with open(File_name, 'rb') as f:
            dataset = pickle.load(f)
        test_datasets.append(dataset)

# %%

# Note that train system is changed here!
train_system='118'


# HeteroGNN
modeltype = 'HetEGAT'
for seed in range(0,Num_seeds):

    save_clf_model = None 
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{train_system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    # generalization
    case_study_path = Folder+f'run_results/cs4tld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model_path=trained_model_path,
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=True)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    #New training 
    case_study_path = Folder+f'run_results/cs4ld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model_path=None,
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=False)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)




# # ===========

modeltype = 'HetMPNN'
for seed in range(0,Num_seeds):

    save_clf_model = None 
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{train_system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    # generalization
    case_study_path = Folder+f'run_results/cs4tld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model_path=trained_model_path,
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=True)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    #New training 
    case_study_path = Folder+f'run_results/cs4ld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HeteroGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                trained_model_path=None,
                                 HetGNN_type = modeltype,
                                train_task=task,test_task=task,
                                hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=False)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)



# # %%

# HomoGNN
modeltype = 'HomoEGAT'
for seed in range(0,Num_seeds):

    save_clf_model = None 
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{train_system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    # generalization
    case_study_path = Folder+f'run_results/cs4tld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=trained_model_path,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=True)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # new train
    case_study_path = Folder+f'run_results/cs4ld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=None,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=False)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)




# #=

# # HomoGNN
modeltype = 'HomoGAT'
for seed in range(0,Num_seeds):

    save_clf_model = None 
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{train_system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    # generalization
    case_study_path = Folder+f'run_results/cs4tld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=trained_model_path,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=True)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)

    # new train
    case_study_path = Folder+f'run_results/cs4ld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=None,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=False)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


#=

# HomoGNN
modeltype = 'HomoMPNN'
for seed in range(0,Num_seeds):

    save_clf_model = None 
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{train_system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    # generalization
    case_study_path = Folder+f'run_results/cs4tld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=trained_model_path,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=True)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)

    # new train
    case_study_path = Folder+f'run_results/cs4ld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=None,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=False)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


#=

# HomoGNN
modeltype = 'HomoGCN'
for seed in range(0,Num_seeds):

    save_clf_model = None 
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{train_system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    # generalization
    case_study_path = Folder+f'run_results/cs4tld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=trained_model_path,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=True)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)


    # new train
    case_study_path = Folder+f'run_results/cs4ld_{train_system}_to_{test_system}bus_{task}_{modeltype}_{dataname}_{traintype}_{testtype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'

    clf_res = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list=test_datasets,
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,
                                trained_model_path=None,
                               hyper_param=hyper_param,num_epochs=num_epochs,seed=seed,trnasfer_learning=False)

    with open(case_study_path, 'wb') as f:
        pickle.dump(clf_res['test_res'], f)






