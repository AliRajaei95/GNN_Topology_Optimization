"""
Training-data generation utilities for DC optimal substation switching (DC-OSS).

This module contains random data-generation, contingency sampling,
and dataset-construction pipelines for machine-learning applications in
power-system topology optimization and congestion management.

The implementations include:
    - Randomized load-generation sampling
    - Line-outage and contingency sampling
    - Random transmission-limit and generator-cost perturbations
    - DC-OPF-based operating-point generation
    - DC optimal substation switching (OSS) simulations
    - Congestion-focused dataset filtering
    - Graph-based feature extraction for ML models
    - Training-target generation for topology-control learning

The generated datasets are intended for supervised learning, graph neural
networks (GNNs), and ML-assisted congestion-management studies.

Developed for power-system operation, topology-control,
and machine-learning research using Pyomo and Gurobi optimization frameworks.

Tested with Python 3.12 and Gurobi 10.0.
"""


# %% 
def generate_random_load_topology_line_gen(
    data,
    Nsamples=1000,
    distribution='kumaraswamy',
    lowD=0.7,
    highD=1.3,
    correlation=0.5,
    a=1.6,
    b=2.8,
    N_k=0,
    N_k_prob=[1.0],
    lineLow=1.0,
    LineHigh=1.0,   # random with uniform dist
    genLow=1.0,
    genHigh=1.0,
    seed=0,         # random with uniform dist
):
    """
    Generate randomized load, topology, line-limit, and generator-cost samples.

    This function creates random operating scenarios for training ML models used
    in DC optimal substation switching and congestion-management studies. It
    samples demand values, line outages, line-limit perturbations, and generator
    cost perturbations.

    Parameters
    ----------
    data : dict
        Input system data dictionary.
    Nsamples : int
        Number of random samples to generate.
    distribution : str
        Demand-sampling distribution. Supported options are 'uniform' and
        'kumaraswamy'.
    lowD : float
        Lower demand scaling factor.
    highD : float
        Upper demand scaling factor.
    correlation : float
        Correlation coefficient used in Kumaraswamy sampling.
    a : float
        First Kumaraswamy distribution parameter.
        a=1.6, b=2.8 gives you almost normal dist for each dim
    b : float
        Second Kumaraswamy distribution parameter.
    N_k : int
        Maximum number of simultaneous line outages.
    N_k_prob : list
        Probabilities for 0 to N_k line outages.
    lineLow : float
        Lower scaling factor for randomized line limits.
    LineHigh : float
        Upper scaling factor for randomized line limits.
    genLow : float
        Lower scaling factor for randomized generator costs.
    genHigh : float
        Upper scaling factor for randomized generator costs.
    seed : int
        Random seed for NumPy and Python random generators.

    Returns
    -------
    dict
        Dictionary containing:
            - X_pd : numpy.ndarray
                Randomized demand samples with shape [Nsamples, Nload].
            - line_outage_list : list
                Sampled line-outage sets for each scenario.
            - line_limit_samples : numpy.ndarray
                Randomized line-limit samples.
            - gen_cost_samples : numpy.ndarray
                Randomized generator-cost samples.
    """

    # === Input data
    Bus = data['Bus']
    DemandSet = data['Demandset']
    Pdemand = data['Pdemand']
    D2B = data['D2B']

    Nload = len(data['Demandset'])
    Nbus = len(data['Bus'])
    Nline = len(data['line'])
    Ngen = len(data['G'])

    # === Input checks
    if len(N_k_prob) != N_k + 1:
        raise ValueError("Length of N_k_prob must be N_k + 1")

    if not abs(sum(N_k_prob) - 1.0) < 1e-6:
        raise ValueError("Sum of N_k_prob must be 1.0")

    # === Random seed
    np.random.seed(seed)
    random.seed(seed)

    # === Demand samples
    X_pd = np.zeros((Nsamples, Nload))

    if distribution == 'uniform':
        Xd_d = np.random.uniform(
            low=lowD,
            high=highD,
            size=(Nsamples, Nload),
        )

    if distribution == 'kumaraswamy':
        Nload = len(DemandSet)

        Xd_d = kumaraswamy_montecarlo(
            a=a,
            b=b,
            c=correlation,
            lower_bounds=np.repeat(lowD, Nload),
            upper_bounds=np.repeat(highD, Nload),
            num_samples=Nsamples,
        ).T

    #         print(Xd)

    # === Topology samples
    topology_samples = []

    for sample in range(Nsamples):
        for d_ind, d in enumerate(DemandSet):
            X_pd[sample, d_ind] = Xd_d[sample, d_ind] * Pdemand.loc[d]['Pd']

        # Sample number of line outages according to N_k_prob
        k = random.choices(range(N_k + 1), weights=N_k_prob)[0]

        # Sample k unique lines to be out
        # outage_lines = random.sample(data['line_cont_notradial'], k) if k > 0 else []
        outage_lines = random.choice(data['line_N-k'][k]) if k > 0 else []

        topology_samples.append(outage_lines)

    print('X_pd shape ', X_pd.shape)
    print('topology data with N_k_prob of ', N_k_prob)

    # === Line-limit samples
    line_limit_samples = np.random.uniform(
        low=lineLow,
        high=LineHigh,
        size=(Nsamples, Nline),
    )

    line_limit_samples = line_limit_samples * data['branch1D']['limit'].to_numpy()

    # === Generator-cost samples
    gen_cost_samples = np.random.uniform(
        low=genLow,
        high=genHigh,
        size=(Nsamples, Ngen),
    )

    gen_cost_samples = gen_cost_samples * data['Gen_data']['b'].to_numpy()

    # === Output dictionary
    return {
        'X_pd': X_pd,
        'line_outage_list': topology_samples,
        'line_limit_samples': line_limit_samples,
        'gen_cost_samples': gen_cost_samples,
    }
    
    



# %%
def generate_TrainingData_DC_OSS_AllSplit_Pyomo(
    data,
    X_pd,
    line_outage_list=[],
    line_limit_samples=None,
    gen_cost_samples=None,
    Max_Sw_bus=1,
    CongFilterLimit=0.9,
    KHops=5,
    Force_Split=True,
    black_list_split=None,
):
    """
    Generate ML training data by solving DC-OPF and DC-OSS instances.

    This function first solves a DC-OPF problem for each sampled operating
    condition. Congested samples are then used to evaluate candidate busbar
    splitting actions through the DC optimal substation switching model.
    The resulting dictionary stores node, edge, flow, injection, topology,
    and switching-label information for ML/GNN training.

    CongFilterLimit is used to identify congested lines. Samples with all lines
    below this thereshold are removed, because they have no congestion.

    Parameters
    ----------
    data : dict
        Input system data dictionary.
    X_pd : numpy.ndarray
        Active-demand samples with shape [Nsamples, Ndemand].
    line_outage_list : list
        List of line-outage sets for each sample.
    line_limit_samples : array-like, optional
        Line-limit samples for each scenario.
    gen_cost_samples : array-like, optional
        Generator-cost samples for each scenario.
    Max_Sw_bus : int
        Maximum number of substations allowed to split.
    CongFilterLimit : float
        Loading threshold used to identify congested samples.
    KHops : int
        Number of hops used to select candidate substations around congestion.
    Force_Split : bool
        If True, force each candidate substation to split when generating labels.
    black_list_split : bool, list, or None
        If True, automatically identify infeasible split nodes.
        If list, use the provided list as the blacklist.
        If None or False, no blacklist is used.

    Returns
    -------
    Finaldict : dict
        Training dataset dictionary containing operating-point features,
        graph features, candidate switching nodes, labels, line outages,
        and hyperparameters.
    """

    # === Input data
    Bus = data['Bus']

    gamma = data['gamma']
    ds_loading = data['ds_loading']

    print('X_pd shape is: ', X_pd.shape)

    Nsamples = X_pd.shape[0]
    Nbus = len(data['Bus'])
    Ng = len(data['G'])
    Nd = len(data['Demandset'])
    Nline = len(data['line'])

    # === Input checks and default samples
    if len(line_outage_list) != X_pd.shape[0]:
        print('\n len X_pd and topology list should be equal!\n')
        line_outage_list = [[] for _ in range(Nsamples)]

    if line_limit_samples is None:
        print('we need lilmit samples!')
        # line_limit_samples = [data['branch1D']['limit'].to_numpy() for _ in range(Nsamples) ]
        line_limit_samples = [None for _ in range(Nsamples)]
        line_limits_input = False
    else:
        line_limits_input = True

    if gen_cost_samples is None:
        print('we need gen cost samples!')
        # gen_cost_samples = [ data['Gen_data']['b'].to_numpy() for _ in range(Nsamples)  ]
        gen_cost_samples = [None for _ in range(Nsamples)]

    # === Candidate switching nodes
    radial_nodes = find_radial_end_nodes(data['Lines1D'])

    Nodes_highdegree = [
        b
        for b in data['Bus']
        if data['NumberL2B'][b] >= 4
    ]

    # Nsw = len(Candidate_bus)

    # === Optimization models
    modelDC = Create_DC_OPF_Pyomo(data=data)

    modelOSS = Create_Cong_DC_OSS_Pyomo(
        data=data,
        Max_Sw_bus=Max_Sw_bus,
        Substation_nodes=Nodes_highdegree,
        LineLimit=10.0,
    )   # we relax the line limits to allow for bad switching

    # === Output dictionary
    Finaldict = {
        'Samples': {sample: {} for sample in range(Nsamples)},
        'HyperParameters': {},
        'Nodes_highdegree': Nodes_highdegree,
    }

    Finaldict['HyperParameters'] = {
        'LineLimit': data['LineLimit'],
        'DemFactor': data['DemFactor'],
        'gamma': data['gamma'],
        'ds_loading': data['ds_loading'],
        'MaxSw': Max_Sw_bus,
        'KHops': KHops,
        'CongFilterLimit': CongFilterLimit,
        'line_reactance': data['branch1D']['x'].to_numpy(),
        # 'GridData':data,
    }

    # === Feature arrays
    X_pg = np.zeros((Nsamples, Ng))                         # all Pg
    X_pflow = np.zeros((Nsamples, len(data['line'])))       # all flows of DC OPF
    X_pflow_list = []                                       # a list of numpy arrays for each sample, includes the flows of lines in service.
    X_node_degree = np.zeros((Nsamples, len(Bus)))

    X_pinj = np.zeros((Nsamples, len(Bus)))                 # injections Pg-Pd

    y_pflow_list = []                                       # a list of numpy arrays for each sample, includes the flows of lines after splitting for good candidates.
    line_limit_list = []
    Edge_sender_list = []
    Edge_receiver_list = []

    dual_list = []                                          # a list of dual values for candidate buses
    hops_list = []

    bus_sw_list = []
    bus_sw_indices_list = []                                # a list of lists that shows the indices of candidate buses for switching.

    X_delta = np.zeros((Nsamples, len(Bus)))                # voltage angles

    infeasible_indices = []
    notcongested_indices = []

    line_outage_list_final = []
    line_outage_indices = []

    # ==========================================================================
    # Identify blacklisted split nodes
    # ==========================================================================

    # finding the black list!
    if black_list_split is None or black_list_split is False:
        print('black list not requested')
        black_list_split = []

    elif isinstance(black_list_split, list):
        print('black list is: ', black_list_split)

    elif black_list_split is True:
        black_list_split = []
        sample = 0

        res_DCOPF = Solve_DC_OPF_Pyomo(
            data,
            modelDC,
            Pd_instance=X_pd[sample],
            lines_outage=line_outage_list[sample],
            limit_new=line_limit_samples[sample],
            gen_cost_new=gen_cost_samples[sample],
            cong_limit=CongFilterLimit,
            print_result=False,
        )

        if res_DCOPF == "Infeasible":
            sample += 1

        else:
            candidate_nodes = Nodes_highdegree

            for b_split in tqdm(candidate_nodes):

                res_OSS = Solve_Cong_DC_OSS_Pyomo(
                    data=data,
                    modelX=modelOSS,
                    Pd_instance=X_pd[sample],
                    lines_outage=line_outage_list[sample],
                    limit_new=line_limit_samples[sample],
                    gen_cost_new=gen_cost_samples[sample],
                    res_DCOPF=res_DCOPF,
                    candidate_bus=[b_split],
                    Substation_nodes=Nodes_highdegree,
                    force_split=[b_split],
                    print_result=False,
                )

                if res_OSS == "Infeasible":   # these nodes are the starting point of radial lines and lead to load shedding and cannot be forced open
                    black_list_split.append(b_split)
                    print('black list updated with ', b_split)

    # ==========================================================================
    # Generate training samples
    # ==========================================================================

    for sample in tqdm(range(Nsamples)):

        # === Solve base DC-OPF
        res_DCOPF = Solve_DC_OPF_Pyomo(
            data,
            modelDC,
            Pd_instance=X_pd[sample],
            lines_outage=line_outage_list[sample],
            limit_new=line_limit_samples[sample],
            gen_cost_new=gen_cost_samples[sample],
            cong_limit=CongFilterLimit,
            print_result=False,
        )

        if res_DCOPF == "Infeasible":
            X_pg[sample] = np.zeros((X_pg.shape[1]))
            infeasible_indices += [sample]
            print('DC OPF infeasible')
            continue

        if res_DCOPF['Num_cong'] < 1:
            X_pg[sample] = -np.ones((X_pg.shape[1]))
            notcongested_indices += [sample]
            continue         # we only want congested lines, this sample will be removed

        # === Candidate nodes around congestion
        candidate_nodes, node_degrees = find_k_hop_nodes_from_congestion(
            data['Bus'],
            data['Lines1D'],
            res_DCOPF['overloaded_mask'],
            lines_outage=line_outage_list[sample],
            k=KHops,
            min_degree=4,
        )   # makes sure even after line outage a node has min 4degree

        candidate_nodes = [
            n
            for n in candidate_nodes
            if n not in black_list_split
        ]   # filter black_list nodes

        X_node_degree[sample, :] = np.array(node_degrees)

        nodes_hops_away = node_distance_from_congestion(
            data['Bus'],
            data['Lines1D'],
            res_DCOPF['overloaded_mask'],
            lines_outage=line_outage_list[sample],
        )

        # === Store base operating-point features
        X_pg[sample] = res_DCOPF['Pg_array']
        X_pflow[sample] = res_DCOPF['Pflow']
        X_pinj[sample] = res_DCOPF['Pinj']
        X_delta[sample] = res_DCOPF['delta']


        # === Prepare label matrix for candidate switching actions
        line_inservice_indices = [
            data['line'].index(l)
            for l in data['line']
            if l not in line_outage_list[sample]
        ]

        y_pflow = np.zeros((len(candidate_nodes), len(line_inservice_indices)))

        OSS_infeasible_splits = []
        rows_to_remove = []

        # if sample not in infeasible_indices+notcongested_indices:
        for b_idx, b_split in enumerate(candidate_nodes):

            if Force_Split == True:

                res_OSS = Solve_Cong_DC_OSS_Pyomo(
                    data=data,
                    modelX=modelOSS,
                    Pd_instance=X_pd[sample],
                    lines_outage=line_outage_list[sample],
                    limit_new=line_limit_samples[sample],
                    gen_cost_new=gen_cost_samples[sample],
                    res_DCOPF=res_DCOPF,
                    candidate_bus=[b_split],
                    Substation_nodes=Nodes_highdegree,
                    force_split=[b_split],
                    print_result=False,
                )

            elif Force_Split == False:

                res_OSS = Solve_Cong_DC_OSS_Pyomo(
                    data=data,
                    modelX=modelOSS,
                    res_DCOPF=res_DCOPF,
                    Pd_instance=X_pd[sample],
                    lines_outage=line_outage_list[sample],
                    limit_new=line_limit_samples[sample],
                    gen_cost_new=gen_cost_samples[sample],
                    candidate_bus=[b_split],
                    Substation_nodes=Nodes_highdegree,
                    force_split=[],
                    print_result=False,
                )

            if res_OSS == "Infeasible":
                # infeasible_indices += [sample]  #this should not happen
                OSS_infeasible_splits.append(b_split)
                rows_to_remove.append(b_idx)
                print(b_split, ' OSS infeasible')
                continue   # go to next sample

            else:
                y_pflow[candidate_nodes.index(b_split)] = res_OSS['Pflow_top']

        # remove OSS infeasible points
        candidate_nodes = [
            b
            for b in candidate_nodes
            if b not in OSS_infeasible_splits
        ]

        y_pflow = np.delete(y_pflow, rows_to_remove, axis=0)

        # add list features
        if sample not in infeasible_indices + notcongested_indices:

            y_pflow_list.append(y_pflow)

            bus_sw_list.append(candidate_nodes)
            bus_sw_indices_list.append([Bus.index(b) for b in candidate_nodes])

            if line_limits_input == True:
                line_limit_list.append(
                    line_limit_samples[sample][line_inservice_indices])   # limit for lines in serice

            else:
                line_limit_list.append(
                    data['branch1D']['limit'].to_numpy()[line_inservice_indices])

            hops_list.append(nodes_hops_away)

            X_pflow_list += [res_DCOPF['Pflow_top']]

            sample_sender = []
            sample_receiver = []

            for l, i, j in data['Lines1D']:
                if l not in line_outage_list[sample]:
                    sample_sender += [data['Bus'].index(i)]
                    sample_receiver += [data['Bus'].index(j)]

            Edge_sender_list += [np.array(sample_sender)]
            Edge_receiver_list += [np.array(sample_receiver)]

            line_outage_list_final.append(line_outage_list[sample])
            line_outage_indices.append(
                [data['line'].index(l) for l in line_outage_list[sample]])

    # ==========================================================================
    # Remove infeasible and non-congested samples
    # ==========================================================================

    print('%i infeasible loaded instances removed.' % len(infeasible_indices))
    print('%i low congestion instances removed.' % len(notcongested_indices))

    X_pg = np.delete(
        X_pg,
        infeasible_indices + notcongested_indices,
        axis=0,
    )

    X_pd = np.delete(
        X_pd,
        infeasible_indices + notcongested_indices,
        axis=0,
    )

    X_pflow = np.delete(
        X_pflow,
        infeasible_indices + notcongested_indices,
        axis=0,
    )

    X_pinj = np.delete(
        X_pinj,
        infeasible_indices + notcongested_indices,
        axis=0,
    )

    # y_pflow = np.delete(y_pflow, infeasible_indices+notcongested_indices, axis=0)

    X_delta = np.delete(
        X_delta,
        infeasible_indices + notcongested_indices,
        axis=0,
    )

    X_node_degree = np.delete(
        X_node_degree,
        infeasible_indices + notcongested_indices,
        axis=0,
    )

    print('%i samples ready.' % X_pg.shape[0])

    # ==========================================================================
    # Store final dataset
    # ==========================================================================

    Finaldict['X_pg'] = X_pg                         # [sample,G]
    Finaldict['X_pd'] = X_pd
    Finaldict['X_pflow_list'] = X_pflow_list         # [sample,line] DC flows
    Finaldict['X_pflow'] = X_pflow
    Finaldict['y_pflow_list'] = y_pflow_list
    Finaldict['X_pinj'] = X_pinj
    Finaldict['line_outage_list'] = line_outage_list_final
    Finaldict['line_outage_indices'] = line_outage_indices
    Finaldict['X_delta'] = X_delta
    Finaldict['Edge_sender_list'] = Edge_sender_list
    Finaldict['Edge_receiver_list'] = Edge_receiver_list
    Finaldict['bus_sw_list'] = bus_sw_list
    Finaldict['bus_sw_indices_list'] = bus_sw_indices_list
    Finaldict['X_node_degree'] = X_node_degree
    Finaldict['line_limit_list'] = line_limit_list
    Finaldict['hops_list'] = hops_list

    return Finaldict