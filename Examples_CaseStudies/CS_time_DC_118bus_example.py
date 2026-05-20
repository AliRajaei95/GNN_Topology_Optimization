"""
Section IV-E: Computational-efficiency case study for ML-assisted topology
control (DC formulation).

This script reproduces the computational-efficiency experiments presented
in Section IV-E of the paper for the DC optimal substation switching (DC-OSS)
framework. The case study evaluates the runtime performance of machine-learning-
assisted topology-control methods compared to optimization-based approaches.

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


# %% Run here

gamma=2
hops=5
Num_seeds=1
layer=5
hid=64
num_epochs=100

hyper_param = { }




# %%
traintype='N0'
system='118'
dataname='dataset3_gen' 

lm=0.8
gamma=2
topk = 5
networkdata = read_data_AC( Folder+f'IEEE_{system}_bus_Data_PGLib_ACOPF.xlsx',DemFactor=1,LineLimit=lm,ds_loading=0.8,Seg_num=5,gamma=gamma)


train_datasets = []
Nkname = 'N-0_[1]'


for seed in range(0,1):
    if system == '118' and dataname == 'dataset3_gen':
        File_name = DataFolder + f'Datasets/118-bus/{dataname}/Dataset_OSS_118bus_{Nkname}_1000samples_0.8-1.2demand_0.8-0.8limit_0.8-1.2gen_0.95congfilter_10hops_seed{seed}.pkl'

    with open(File_name, 'rb') as f:
        dataset = pickle.load(f)

    train_datasets.append(dataset)



# HomoGNN
modeltype = 'HomoMPNN'
for seed in range(0,Num_seeds):

    case_study_path = Folder+f'run_results/cs5_{system}bus_ClfReg_{modeltype}_{topk}top_{traintype}_to_same_{dataname}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_seed{seed}.pkl'
    
    task = 'Clf'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    Clfres = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=None,trained_model_path=trained_model_path)
    
    task = 'Reg'
    trained_model_path = Folder+f'saved_models/{modeltype}_{task}_{system}bus_{dataname}_{traintype}_gamma{gamma}_hops{hops}_layer{layer}_hid{hid}_epoch{num_epochs}_seed{seed}.pth'

    Regres = run_casestudy_HomoGNN(train_dataset_list=train_datasets,
                                test_dataset_list='same',
                                HomoGNN_type=modeltype,
                                train_task=task,test_task=task,hyper_param=hyper_param,num_epochs=0,seed=seed,
                                save_model_path=None,trained_model_path=trained_model_path)
    
    
    cs_res = CaseStudy5_Time_Clf_Reg(networkdata,Clfres['test_res'],Regres['test_res'],
                                     modeltype=modeltype,TopMLActions=topk  )

    with open(case_study_path, 'wb') as f:
        pickle.dump(cs_res, f)

    