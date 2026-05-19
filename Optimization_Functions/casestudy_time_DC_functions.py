"""
Utility functions for the Section IV-E computational-efficiency case study.

This module contains helper functions used to evaluate the computational
performance of machine-learning-assisted topology-control methods within
the DC optimal substation switching (DC-OSS) framework.

The implementations include:
    - Runtime evaluation for ML-assisted DC-OSS
    - Comparison of optimization-based and ML-guided switching strategies
    - Classification- and regression-based candidate selection
    - Congestion-cost evaluation
    - Random AC load generation for runtime studies
    - Candidate-substation filtering using K-hop congestion regions
    - Runtime benchmarking for different GNN architectures

The functions are used to reproduce the computational-efficiency
experiments presented in Section IV-E of the paper.
"""

def CaseStudy5_Time_Clf_Reg(networkdata,CLf_test_res,Reg_test_res,modeltype='FNN',
                    Max_Sw_bus=1,TopMLActions=5,case_study_path=None):
    
    """
    
    """
    

    Bus = networkdata['Bus']
    gamma = networkdata['gamma']
    ds_loading = networkdata['ds_loading'] 

    # with open(test_data_path, 'rb') as f:
    #         test_res = pickle.load(f)


    system_size = len(networkdata['Bus'])
    num_samples = len(CLf_test_res['input_by_sample'])

    if len(CLf_test_res['input_by_sample']) != len(Reg_test_res['input_by_sample']):
        print('Error!\nClf and Reg are not the same!')
        return False

    X_inj = np.zeros((num_samples,system_size))
    mask_arr = np.zeros((num_samples,system_size), dtype=bool)
    bus_sw_list = [] 
    


    if modeltype == 'FNN':
        print('model is FNN')
        for i in range(num_samples):
            X_inj[i,:] = CLf_test_res['input_by_sample'][i][:system_size*3].reshape(system_size,-1)[:,0] #*3 bus features
            mask = CLf_test_res['mask_by_sample'][i] 
            mask_arr[i,:] = mask
            selected_buses = list(compress(Bus, mask))
            bus_sw_list.append(selected_buses)
    elif modeltype == 'HomoMPNN' or modeltype=='HomoEGAT' or modeltype=='HomoGNN':
        print('model is HomoGNN')
        for i in range(num_samples):
            X_inj[i,:] = CLf_test_res['input_by_sample'][i][:,0] #first feature is inj, 
            mask = CLf_test_res['mask_by_sample'][i]
            mask_arr[i,:] = mask
            selected_buses = list(compress(Bus, mask))
            bus_sw_list.append(selected_buses)
            


    print('sum X_inj is %.5f' %X_inj.sum())


    #find ML candidates
    Clf_candidate_subs = []
    Reg_candidate_subs = []



    for i in range(num_samples):
      # if task == 'Clf':
      # Clf
      pred = CLf_test_res['target_by_sample'][i][ mask_arr[i] ]
      paired = [(p, b) for p, b in zip(pred, bus_sw_list[i]) if p > -0.5]
      paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
      top_k = paired_sorted[:TopMLActions]
      top_k_buses = [b for _, b in top_k]
      Clf_candidate_subs.append(top_k_buses)

      # if task == 'Reg':
      #Reg
      pred = Reg_test_res['target_by_sample'][i][ mask_arr[i] ]
      paired = [(p, b) for p, b in zip(pred, bus_sw_list[i]) if p > -0.5]
      paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
      top_k = paired_sorted[:TopMLActions]
      top_k_buses = [b for _, b in top_k]
      Reg_candidate_subs.append(top_k_buses)




       

    Nodes_highdegree = [b for b in networkdata['Bus'] if networkdata['NumberL2B'][b]>=4]


    modelOSS_ML= Create_ML_Cong_DC_OSS_Pyomo(networkdata,Max_Sw_bus=Max_Sw_bus,
                                             Substation_nodes=Nodes_highdegree,
                                               LineLimit=1) #we don't force splitting here
    
    modelOPF= Create_ML_Cong_DC_OSS_Pyomo(networkdata,Max_Sw_bus=0,
                                             Substation_nodes=[] ,
                                               LineLimit=1) #we don't force splitting here

    
    #TODO: maybe we have to create the model for each set of candidate solutions for better timing?
    
    cases = ['All_subs','Khops','Clf','Reg','NoSW']
    metrics = ['Pflow','CongCost','time']
    Finaldict = {c: {m:{} for m in metrics     }   for c in cases  }

    inf_samples = []
    for sample in tqdm(range(num_samples)):
        
        # All_subs the ground truth ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    X_inj=X_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Nodes_highdegree, 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['All_subs']['Pflow'][sample] = res['Pflow']
        Finaldict['All_subs']['CongCost'][sample] = res['CongCost']
        Finaldict['All_subs']['time'][sample] = res['time']



        # Khops  ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    X_inj=X_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=bus_sw_list[i], 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['Khops']['Pflow'][sample] = res['Pflow']
        Finaldict['Khops']['CongCost'][sample] = res['CongCost']
        Finaldict['Khops']['time'][sample] = res['time']

        # print('Khops %.3f'%res['CongCost'])


        # Clf  ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    X_inj=X_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Clf_candidate_subs[i], 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['Clf']['Pflow'][sample] = res['Pflow']
        Finaldict['Clf']['CongCost'][sample] = res['CongCost']
        Finaldict['Clf']['time'][sample] = res['time']


        # Reg  ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    X_inj=X_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Reg_candidate_subs[i], 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['Reg']['Pflow'][sample] = res['Pflow']
        Finaldict['Reg']['CongCost'][sample] = res['CongCost']
        Finaldict['Reg']['time'][sample] = res['time']
      


        # NoSW ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOPF,
                                    X_inj=X_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=[], 
                                    Substation_nodes=[],
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['NoSW']['Pflow'][sample] = res['Pflow']
        Finaldict['NoSW']['CongCost'][sample] = res['CongCost']
        Finaldict['NoSW']['time'][sample] = res['time']


        if case_study_path is not None:
            with open(case_study_path, 'wb') as f:
                pickle.dump(Finaldict, f)

        
    Finaldict['inf_samples'] = inf_samples
    
    return Finaldict




# %%

def CaseStudy5_DC_Time_NoML(networkdata,num_samples=10,Max_Sw_bus=1,
                            lowD=0.8,highD=1.2,case_study_path=None):
    
    """

    """
    

    Bus = networkdata['Bus']
    gamma = networkdata['gamma']
    ds_loading = networkdata['ds_loading'] 

    

    system_size = len(networkdata['Bus'])

    res = generate_random_ac_load(data=networkdata,Nsamples=num_samples,distribution='kumaraswamy',
                        lowD=lowD,highD=highD,correlation=0.5,a=1.6,b=2.8,seed=0)
    
    X_pd = res['X_pd']

    


    # X_inj = np.zeros((num_samples,system_size))
    P_inj = np.zeros((num_samples,system_size))    

       

    Nodes_highdegree = [b for b in networkdata['Bus'] if networkdata['NumberL2B'][b]>=4]


    modelDCOPF0 = Create_DC_OPF_Pyomo(networkdata)


    modelOSS= Create_ML_Cong_DC_OSS_Pyomo(networkdata,Max_Sw_bus=Max_Sw_bus,
                                             Substation_nodes=Nodes_highdegree,
                                               LineLimit=1) #we don't force splitting here
    
    modelOPF= Create_ML_Cong_DC_OSS_Pyomo(networkdata,Max_Sw_bus=0,
                                             Substation_nodes=[] ,
                                               LineLimit=1) #we don't force splitting here

    
    
    cases = ['All_subs','Khops','NoSW']
    metrics = ['Pflow','CongCost','time']
    Finaldict = {c: {m:{} for m in metrics     }   for c in cases  }

    

    inf_samples = []
    nocong_sample = []

    for sample in tqdm(range(num_samples)):


        res = Solve_DC_OPF_Pyomo(networkdata,modelDCOPF0,
                                       Pd_instance=X_pd[sample,:],
                                       cong_limit=0.95,
                                 print_result=False)
        # print(res['Num_cong'])

        if res == "Infeasible":
            inf_samples += [sample]
            print('sample %i infeasible'%sample)
            continue
        
        if res['Num_cong'] < 1:
           print('sample %i non congested'%sample)
           nocong_sample += [sample]
           continue  
        
        
        
        P_inj[sample,:] = res['Pinj']

        khop_nodes, node_degrees = find_k_hop_nodes_from_congestion(networkdata['Bus'],networkdata['Lines1D'], res['overloaded_mask'], lines_outage=[],
                                                                          k=5, min_degree=4) #makes sure even after line outage a node has min 4degree
        
        khop_nodes = [n for n in khop_nodes ] #filter black_list nodes

    
        # # NoSw  ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOPF,
                                    X_inj=P_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=[], 
                                    Substation_nodes=[],
                                    print_result=False)
        if res == "Infeasible":
            print('sample %i infeasible in NoSw'%sample)
            inf_samples.append(sample)
            continue
        Finaldict['NoSW']['Pflow'][sample] = res['Pflow']
        Finaldict['NoSW']['CongCost'][sample] = res['CongCost']
        Finaldict['NoSW']['time'][sample] = res['time']


        
        # All_subs the ground truth ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOSS,
                                    X_inj=P_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Nodes_highdegree, 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            print('sample %i infeasible in All_sub'%sample)
            inf_samples.append(sample)
            continue
        Finaldict['All_subs']['Pflow'][sample] = res['Pflow']
        Finaldict['All_subs']['CongCost'][sample] = res['CongCost']
        Finaldict['All_subs']['time'][sample] = res['time']



        # Khops  ======================================================================
        res = Solve_ML_Cong_DC_OSS_Pyomo(data=networkdata,modelX=modelOSS,
                                    X_inj=P_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=khop_nodes, 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            print('sample %i infeasible in Khops'%sample)
            inf_samples.append(sample)
            continue
        Finaldict['Khops']['Pflow'][sample] = res['Pflow']
        Finaldict['Khops']['CongCost'][sample] = res['CongCost']
        Finaldict['Khops']['time'][sample] = res['time']

        # print('Khops %.3f'%res['CongCost'])

        
        if case_study_path is not None:
            with open(case_study_path, 'wb') as f:
                pickle.dump(Finaldict, f)

        
    Finaldict['inf_samples'] = inf_samples
    Finaldict['nocong_samples'] = nocong_sample

    return Finaldict



def CaseStudy5_AC_Time_Clf_Reg(networkdata,CLf_test_res,Reg_test_res,modeltype='FNN',
                    Max_Sw_bus=1,TopMLActions=5,PQfactor = 0.3,case_study_path=None):
    
    """
    
    """
    

    Bus = networkdata['Bus']
    gamma = networkdata['gamma']
    ds_loading = networkdata['ds_loading'] 

    # with open(test_data_path, 'rb') as f:
    #         test_res = pickle.load(f)


    system_size = len(networkdata['Bus'])
    num_samples = len(CLf_test_res['input_by_sample'])

    if len(CLf_test_res['input_by_sample']) != len(Reg_test_res['input_by_sample']):
        print('Error!\nClf and Reg are not the same!')
        return False

    X_inj = np.zeros((num_samples,system_size))
    P_inj = np.zeros((num_samples,system_size))
    Q_inj = np.zeros((num_samples,system_size))
    



    mask_arr = np.zeros((num_samples,system_size), dtype=bool)
    bus_sw_list = [] 
    


    if modeltype == 'FNN':
        print('model is FNN')
        for i in range(num_samples):
            X_inj[i,:] = CLf_test_res['input_by_sample'][i][:system_size*3].reshape(system_size,-1)[:,0] #*3 bus features
            mask = CLf_test_res['mask_by_sample'][i] 
            mask_arr[i,:] = mask
            selected_buses = list(compress(Bus, mask))
            bus_sw_list.append(selected_buses)
    elif modeltype == 'HomoMPNN' or modeltype=='HomoEGAT' or modeltype=='HomoGNN':
        print('model is HomoGNN')
        for i in range(num_samples):
            X_inj[i,:] = CLf_test_res['input_by_sample'][i][:,0] #first feature is inj, 
            mask = CLf_test_res['mask_by_sample'][i]
            mask_arr[i,:] = mask
            selected_buses = list(compress(Bus, mask))
            bus_sw_list.append(selected_buses)
            


    print('sum X_inj is %.5f' %X_inj.sum())



    #find ML candidates
    Clf_candidate_subs = []
    Reg_candidate_subs = []



    for i in range(num_samples):
      # if task == 'Clf':
      # Clf
      pred = CLf_test_res['target_by_sample'][i][ mask_arr[i] ]
      paired = [(p, b) for p, b in zip(pred, bus_sw_list[i]) if p > -0.5]
      paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
      top_k = paired_sorted[:TopMLActions]
      top_k_buses = [b for _, b in top_k]
      Clf_candidate_subs.append(top_k_buses)

      # if task == 'Reg':
      #Reg
      pred = Reg_test_res['target_by_sample'][i][ mask_arr[i] ]
      paired = [(p, b) for p, b in zip(pred, bus_sw_list[i]) if p > -0.5]
      paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
      top_k = paired_sorted[:TopMLActions]
      top_k_buses = [b for _, b in top_k]
      Reg_candidate_subs.append(top_k_buses)




       

    Nodes_highdegree = [b for b in networkdata['Bus'] if networkdata['NumberL2B'][b]>=4]


    modelACfeasible = Create_AC_feasible_OPF(networkdata)




    modelOSS_ML= Create_ML_Cong_AC_OSS_Pyomo(networkdata,Max_Sw_bus=Max_Sw_bus,
                                             Substation_nodes=Nodes_highdegree,
                                               LineLimit=1) #we don't force splitting here
    
    modelOPF= Create_ML_Cong_AC_OSS_Pyomo(networkdata,Max_Sw_bus=0,
                                             Substation_nodes=[] ,
                                               LineLimit=1) #we don't force splitting here

    
    #TODO: maybe we have to create the model for each set of candidate solutions for better timing?
    
    cases = ['All_subs','Khops','Clf','Reg','NoSW']
    metrics = ['Pflow','CongCost','time']
    Finaldict = {c: {m:{} for m in metrics     }   for c in cases  }

    

    inf_samples = []

    for sample in tqdm(range(num_samples)):
        

        res = Solve_AC_feasible_OPF(data=networkdata,modelX=modelACfeasible,
                                    X_Pinj=X_inj[sample,:], 
                                    X_Qinj=PQfactor*X_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        
        
        P_inj[sample,:] = res['pinj']
        Q_inj[sample,:] = res['qinj']


        
        # All_subs the ground truth ======================================================================
        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Nodes_highdegree, 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['All_subs']['Pflow'][sample] = res['Pflow']
        Finaldict['All_subs']['CongCost'][sample] = res['CongCost']
        Finaldict['All_subs']['time'][sample] = res['time']



        # Khops  ======================================================================
        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=bus_sw_list[i], 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['Khops']['Pflow'][sample] = res['Pflow']
        Finaldict['Khops']['CongCost'][sample] = res['CongCost']
        Finaldict['Khops']['time'][sample] = res['time']

        # print('Khops %.3f'%res['CongCost'])


        # Clf  ======================================================================
        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Clf_candidate_subs[i], 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['Clf']['Pflow'][sample] = res['Pflow']
        Finaldict['Clf']['CongCost'][sample] = res['CongCost']
        Finaldict['Clf']['time'][sample] = res['time']


        # Reg  ======================================================================
        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOSS_ML,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=Reg_candidate_subs[i], 
                                    Substation_nodes=Nodes_highdegree,
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['Reg']['Pflow'][sample] = res['Pflow']
        Finaldict['Reg']['CongCost'][sample] = res['CongCost']
        Finaldict['Reg']['time'][sample] = res['time']
      


        # NoSW ======================================================================
        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOPF,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=[], 
                                    Substation_nodes=[],
                                    print_result=False)
        if res == "Infeasible":
            inf_samples.append(sample)
            continue
        Finaldict['NoSW']['Pflow'][sample] = res['Pflow']
        Finaldict['NoSW']['CongCost'][sample] = res['CongCost']
        Finaldict['NoSW']['time'][sample] = res['time']

        # print(res['Sflow'])
        # print(res['Pflow']**2 + res['Qflow']**2)

        if case_study_path is not None:
            with open(case_study_path, 'wb') as f:
                pickle.dump(Finaldict, f)

        
    Finaldict['inf_samples'] = inf_samples

    return Finaldict



# %%



def CaseStudy5_AC_Time_NoML(networkdata,num_samples=10,Max_Sw_bus=1,
                            lowD=0.8,highD=1.2,case_study_path=None):
    
    """

    """
    

    Bus = networkdata['Bus']
    gamma = networkdata['gamma']
    ds_loading = networkdata['ds_loading'] 

    # with open(test_data_path, 'rb') as f:
    #         test_res = pickle.load(f)


    system_size = len(networkdata['Bus'])

    res = generate_random_ac_load(data=networkdata,Nsamples=num_samples,distribution='kumaraswamy',
                        lowD=lowD,highD=highD,correlation=0.5,a=1.6,b=2.8,seed=0)
    
    X_pd = res['X_pd']
    X_qd = res['X_qd']

    
    P_inj = np.zeros((num_samples,system_size))
    Q_inj = np.zeros((num_samples,system_size))
    
       

    Nodes_highdegree = [b for b in networkdata['Bus'] if networkdata['NumberL2B'][b]>=4]


    modelACOPF0 = Create_AC_OPF_Pyomo(networkdata)


    # modelOSS= Create_ML_Cong_AC_OSS_Pyomo(networkdata,Max_Sw_bus=Max_Sw_bus,
    #                                          Substation_nodes=Nodes_highdegree,
    #                                            LineLimit=1) #we don't force splitting here
    
    modelOPF= Create_ML_Cong_AC_OSS_Pyomo(networkdata,Max_Sw_bus=0,
                                             Substation_nodes=[] ,
                                               LineLimit=1) #we don't force splitting here

    
    
    cases = ['All_subs','Khops','NoSW']
    metrics = ['Pflow','CongCost','time','Sflow']
    Finaldict = {c: {m:{} for m in metrics     }   for c in cases  }

    

    inf_samples = []
    nocong_sample = []

    for sample in tqdm(range(num_samples)):


        res = Solve_AC_OPF_Pyomo(networkdata,modelACOPF0,
                                       Pd_instance=X_pd[sample,:],Qd_instance=X_qd[sample,:],
                                       cong_limit=0.95,
                                 print_result=False)
        # print(res['Num_cong'])

        if res == "Infeasible":
            inf_samples += [sample]
            print('sample %i infeasible'%sample)
            continue
        
        if res['Num_cong'] < 1:
           print('sample %i non congested'%sample)
           nocong_sample += [sample]
           continue  
        
        
        
        P_inj[sample,:] = res['Pinj']
        Q_inj[sample,:] = res['Qinj']

        khop_nodes, node_degrees = find_k_hop_nodes_from_congestion(networkdata['Bus'],networkdata['Lines1D'], res['overloaded_mask'], lines_outage=[],
                                                                          k=5, min_degree=4) #makes sure even after line outage a node has min 4degree
        
        khop_nodes = [n for n in khop_nodes ] #filter black_list nodes


        # # NoSw  ======================================================================
        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOPF,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=[], 
                                    Substation_nodes=[],
                                    print_result=False)
        if res == "Infeasible":
            print('sample %i infeasible in NoSw'%sample)
            inf_samples.append(sample)
            continue
        Finaldict['NoSW']['Sflow'][sample] = res['Sflow']
        Finaldict['NoSW']['CongCost'][sample] = res['CongCost']
        Finaldict['NoSW']['time'][sample] = res['time']


        # Khops  ======================================================================
        modelOSS= Create_ML_Cong_AC_OSS_Pyomo(networkdata,Max_Sw_bus=Max_Sw_bus,
                                             Substation_nodes=khop_nodes,
                                               LineLimit=1) #we don't force splitting here
        

        res = Solve_ML_Cong_AC_OSS_Pyomo(data=networkdata,modelX=modelOSS,
                                    P_inj=P_inj[sample,:],
                                    Q_inj=Q_inj[sample,:],
                                    lines_outage=[],    #not modeled here
                                    candidate_bus=khop_nodes, 
                                    Substation_nodes=khop_nodes,
                                    print_result=False)
        if res == "Infeasible":
            print('sample %i infeasible in Khops'%sample)
            inf_samples.append(sample)
            continue
        Finaldict['Khops']['Sflow'][sample] = res['Sflow']
        Finaldict['Khops']['CongCost'][sample] = res['CongCost']
        Finaldict['Khops']['time'][sample] = res['time']

        
        if case_study_path is not None:
            with open(case_study_path, 'wb') as f:
                pickle.dump(Finaldict, f)


        
    Finaldict['inf_samples'] = inf_samples
    Finaldict['nocong_samples'] = nocong_sample

    return Finaldict




def generate_random_ac_load(data,Nsamples=1000,distribution='kumaraswamy',
                        lowD=0.7,highD=1.3,correlation=0.5,a=1.6,b=2.8,
                       seed=0): #random with uniform dist
    """
    return numpy_array [Nbus,D]
        
    """
        
    DemandSet = data['Demandset'] 
    Pdemand = data['Pdemand']
    Nload=len(data['Demandset'])
  
        
    np.random.seed(seed)
    random.seed(seed)

    X_pd = np.zeros((Nsamples,Nload))
    X_qd = np.zeros((Nsamples,Nload))
    
    if distribution == 'uniform':
        Xd_d=np.random.uniform(low=lowD,high=highD,size=(Nsamples,Nload))

        
        
    if distribution == 'kumaraswamy':
        Nload = len(DemandSet)
        
        Xd_d = kumaraswamy_montecarlo(a=a, b=b, c=correlation, 
                                    lower_bounds=np.repeat(lowD,Nload), upper_bounds=np.repeat(highD,Nload),num_samples=Nsamples).T
    
#         print(Xd)
    
    topology_samples = []
    
    for sample in range(Nsamples):
        for d_ind,d in enumerate(DemandSet):
            X_pd[sample,d_ind] = Xd_d[sample,d_ind]*Pdemand.loc[d]['Pd']
            X_qd[sample,d_ind] = Xd_d[sample,d_ind]*Pdemand.loc[d]['Qd']


        
            
    print('X_pd shape generated ',X_pd.shape)


    
    return {'X_pd':X_pd,
            'X_qd':X_qd}
    