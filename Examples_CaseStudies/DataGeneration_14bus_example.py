"""
Example workflow for generating DC-OSS machine-learning training datasets.

This example demonstrates the complete pipeline for generating randomized
power-system operating conditions and creating supervised learning datasets
for optimal substation switching (OSS) and congestion-management studies.

"""

# %%
exec(open('loadsampling.py').read())
exec(open('DC_Optimal_Splitting_Functions_Pyomo.py').read())
exec(open('DataGeneration_OSS.py').read())


# %% [markdown]
# # Random Load

# %%
sys='14'
Nsamples = 10000        #6hr
lowD = 1.0; highD = 2.5
lineLow=0.5; lineHigh=1.0
genLow=0.8; genHigh=1.2
cong = 0.98
N_k = 1
N_k_prob = [0.5,0.5]
KHops=5


data = read_data_AC(f'IEEE_{sys}_bus_Data_PGLib_ACOPF.xlsx',gamma=4,Seg_num=5,ds_loading=0.8)



result = generate_random_load_topology_line_gen(data=data,Nsamples=Nsamples,
                                                lowD=lowD,highD=highD,
                                                lineLow=lineLow, LineHigh=lineHigh,
                                                genLow=genLow,genHigh=genHigh,
                                       N_k=N_k,N_k_prob=N_k_prob)

X_pd , line_outage_list, line_limit_samples, gen_cost_samples = result['X_pd'], result['line_outage_list'], result['line_limit_samples'], result['gen_cost_samples']


TrainingData = generate_TrainingData_DC_OSS_AllSplit_Pyomo(data=data,X_pd=X_pd,line_outage_list=line_outage_list,
                                                           line_limit_samples=line_limit_samples,
                                                           gen_cost_samples=gen_cost_samples,
                                                           CongFilterLimit=cong,KHops=KHops, Force_Split=True)


File_name = f'Datasets/Dataset_OSS_{sys}bus_N-{N_k}_{N_k_prob}_{Nsamples}samples_{lowD}-{highD}demand_{lineLow}-{lineHigh}limit_{genLow}-{genHigh}gen_{cong}congfilter.pkl'

with open(File_name, 'wb') as f:
    pickle.dump(TrainingData, f)

