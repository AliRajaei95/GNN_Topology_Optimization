"""
DC-based optimal substation switching and congestion-management utilities.

This module contains data-loading helpers, graph/network utilities,
DC-OPF formulations, and Pyomo-based DC optimal substation switching (OSS)
models for congestion management and topology optimization in power systems.

The implementations include:
    - preparing input grid data
    - DC optimal power flow (DC-OPF)
    - Congestion-management OSS formulations
    - Machine-learning-assisted DC-OSS models

Developed for power-system operation, congestion management,
and topology-control studies using Pyomo and Gurobi optimization frameworks.

Tested with Python 3.12 and Gurobi 12.0.
"""

import itertools
import logging
import time
import gurobipy as gp
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import Constraint, ConstraintList, SolverFactory
from pyomo.opt import SolverStatus


# %% input grid data
def read_data_AC(
    File='IEEE_14_bus_Data.xlsx',
    print_data=False,
    LineLimit=1.0,
    DemFactor=1.0,
    remove_line=[],
    Seg_num=10,
    gamma=5,
    ds_loading=0.9,
    Sbase=100,
    N_k=1,
):
    """
    Read network, demand, generation, and congestion-segment data from an Excel file.

    This function prepares all input data required for AC/DC optimal power flow and
    busbar-splitting studies. It reads buses, branches, generators, demands, line limits,
    contingency sets, graph connectivity, and congestion-cost segments, then stores them
    in a dictionary used by the optimization models.

    Parameters
    ----------
    File : str
        Excel input data file.
    print_data : bool
        If True, display the demand, branch, and generation dataframes.
    LineLimit : float
        Scaling factor applied to transmission line limits.
    DemFactor : float
        Scaling factor applied to active and reactive demand.
    remove_line : list
        List of line IDs to remove from the network.
    Seg_num : int
        Number of segments used for piecewise-linear congestion cost.
    gamma : float
        Congestion-cost exponent/weighting parameter.
    ds_loading : float
        Loading threshold used to define congestion segments.
    Sbase : float
        System base power in MVA.
    N_k : int
        Maximum number of simultaneous line outages considered.

    Returns
    -------
    data : dict
        Dictionary containing sets, parameters, network data, graph objects,
        contingency sets, and congestion-cost data.
    """

    data = {}

    # ============================ Bus and busbar sets ============================

    Bus = pd.read_excel(File, sheet_name='Bus', skiprows=0, index_col=[0], usecols='A')
    Bus = list(Bus.index)                   # for b in Bus
    busbar = ['busbar1', 'busbar2']         # set of busbars at each substation, # for i in busbar

    ##============================ Branch bi-directional ========================

    branch = pd.read_excel(File, sheet_name='Branch', skiprows=1, index_col=[0, 1, 2], usecols='A:G')
    if remove_line != []:
        print('\n removing lines: ', remove_line)
        branch.drop(remove_line, axis=0, level=0, inplace=True)
    line = pd.read_excel(File, sheet_name='Branch', skiprows=1, index_col=0, usecols='A')

    if remove_line != []:
        line.drop(remove_line, axis=0, inplace=True)
    line = list(line.index)                 # line is not bi-directional
    branch1D = branch.copy()

    # Add the reverse direction for each branch to create a bi-directional line set.
    for l, i, j in branch.index:
        branch.loc[(l, j, i)] = branch.loc[(l, i, j)]   # bi-directional

    # Convert line limits to per-unit and compute admittance-related branch parameters.
    branch['limit'] = (LineLimit * branch['limit']) / Sbase
    branch['z2'] = branch['r'] ** 2 + branch['x'] ** 2
    branch['g_ij'] = branch['r'] / branch['z2']
    branch['b_ij'] = -branch['x'] / branch['z2']

    # same stuff
    branch1D['limit'] = (LineLimit * branch1D['limit']) / Sbase
    branch1D['z2'] = branch1D['r'] ** 2 + branch1D['x'] ** 2
    branch1D['g_ij'] = branch1D['r'] / branch1D['z2']
    branch1D['b_ij'] = -branch1D['x'] / branch1D['z2']


    br_list = list(branch.index)
    Lines = gp.tuplelist(br_list)           # Lines is a gurobipy tupilelist that shows CONECCTIVITY of the network

    Lines1D = gp.tuplelist(list(branch1D.index))   # 1-directional Lines

    # ============================ Line-to-bus connectivity ============================

    L2B = []                                # set all lines connected to bus b
    for l, i, j in Lines:
        L2B = L2B + [(l, i)]
    L2B = gp.tuplelist(L2B)

    NumberL2B = dict.fromkeys(Bus)          # number of connected lines to each bus (substation)
    for i in NumberL2B.keys():
        n = 0
        NumberL2B[i] = n
        for l in L2B.select('*', i):
            n += 1
            NumberL2B[i] = n



    # Demand Set
    Pdemand = pd.read_excel(File, sheet_name='DemandSet', skiprows=1, index_col=2, usecols='A:F')
    Pdemand.drop(columns=['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)
    Pdemand.rename(columns={1: 'Pd'})
    Pdemand.rename(columns={2: 'Qd'})
    Pdemand.fillna(0, inplace=True)

    # Convert demand to per-unit values.
    Pdemand['Pd'] = (DemFactor * Pdemand['Pd']) / Sbase          # pu
    Pdemand['Qd'] = (DemFactor * Pdemand['Qd']) / Sbase          # pu

    DemandSet = list(Pdemand.index)                              # for d in DemandSet

    D2B_df = pd.read_excel(File, sheet_name='D2B', skiprows=0, index_col=[1, 2])
    d2b_list = list(D2B_df.index)
    D2B = gp.tuplelist(d2b_list)

    Pd_nominal = Pdemand['Pd'].to_numpy()

    # make Pd_array with all buses:
    Pd_array = np.zeros((len(Bus)))
    for d in DemandSet:
        for dx, b in D2B.select(d, '*'):
            Pd_array[Bus.index(b)] += Pdemand.loc[d]['Pd']



    # ==================================== Generation data =============
    Gen_data = pd.read_excel(File, sheet_name='Gen', skiprows=2, index_col=0, usecols='A:G')

    # Convert generator limits to per-unit values.
    Gen_data['Pmax'] = Gen_data['Pmax'] / Sbase
    Gen_data['Pmin'] = Gen_data['Pmin'] / Sbase
    Gen_data['Qmax'] = Gen_data['Qmax'] / Sbase
    Gen_data['Qmin'] = Gen_data['Qmin'] / Sbase

    # Reserve cost
    res2prod_coeff = 0.5
    Gen_data['c_res'] = Gen_data['b'] * res2prod_coeff

    G = list(Gen_data.index)               # Set Generation

    # G2B is a GP object showing Gen-2-bus
    G2B_df = pd.read_excel(File, sheet_name='G2B', skiprows=0, index_col=[1, 2])
    g2b_list = list(G2B_df.index)
    G2B = gp.tuplelist(g2b_list)


    # ============================ Contingency and graph data ============================

    line_cont = [l for l in line]

    # radial lines
    radial_lines = Find_radial_lines(Lines, Bus)
    line_cont_notradial = [l for l in line if l not in radial_lines]

    # N-k non-isolating lines
    data['line_N-k'] = {}
    for k in range(0, N_k + 1):
        data['line_N-k'][k] = find_non_radial_Nk(Lines1D, Bus, k)

    graph = Build_graph_default(Bus, Lines)


    # ============================ Congestion-cost segments ============================

    Cong_sg_data, Seg = compute_congestion_segments(
        line,
        branch['limit'].to_numpy(),
        Seg_num,
        ds_loading,
        gamma,
    )

    nodes = pd.read_excel(File, sheet_name='Bus', skiprows=0, index_col=0, usecols='A:E')

    if print_data == True:
        display(Pdemand)
        display(branch)
        display(Gen_data)



    # ============================ Store model data ============================

    data['Sbase'] = Sbase
    data['Max_MIPGap'] = 0.01
    data['Max_timelimit'] = 600   # 900

    # Network sets
    data['Bus'] = Bus             # for b in Bus
    data['busbar'] = busbar
    data['branch'] = branch
    data['branch1D'] = branch1D
    data['Lines'] = Lines
    data['line'] = line
    data['Lines1D'] = Lines1D
    data['L2B'] = L2B
    data['NumberL2B'] = NumberL2B

    # Demand data
    data['Pdemand'] = Pdemand
    data['Demandset'] = DemandSet
    data['D2B'] = D2B
    data['Pd_nominal'] = Pd_nominal
    data['Pd_array'] = Pd_array

    # Generation data
    data['Gen_data'] = Gen_data
    data['G'] = G                 # for g in Gset
    data['G2B'] = G2B

    # Voltage and big-M parameters
    data['vmin'] = 0.9
    data['vmax'] = 1.1
    data['beta'] = 0
    data['BigM_l'] = 2 * branch['limit']
    data['BigM_b'] = (4 * 3.14) / 3
    data['Maxdelta'] = 10
    data['MaxV2'] = 4
    data['BigM_busbar'] = 10


    # Contingency and graph information
    data['line_cont'] = line_cont
    data['line_cont_notradial'] = line_cont_notradial
    data['radial_lines'] = radial_lines
    data['graph'] = graph

    # Congestion objective data
    data['Seg'] = Seg
    data['Seg_num'] = Seg_num
    data['Cong_sg_data'] = Cong_sg_data
    data['gamma'] = gamma
    data['ds_loading'] = ds_loading
    data['LineLimit'] = LineLimit
    data['DemFactor'] = DemFactor
    data['nodes'] = nodes



    return data




#===============
def compute_congestion_segments(line, limits, Seg_num=4, ds_loading=0.8, gamma=4):
    """
    Computes piecewise linear congestion cost for a given line.

    Parameters:
    - line: list of lines
    - limits: flow limit for the line
    - seg_num: number of segments (default 4)
    - ds_loading: desired loading ratio (default 0.8)
    - gamma: exponent for cost function (default 4)

    Returns:
    - DataFrame indexed by segment name with columns:
      'df', 'fini', 'ffin', 'Cini', 'Cfin', 's'
    """
    

    Seg=['sg'+str(n) for n in range(1,Seg_num+1)]
    seg_ind=pd.MultiIndex.from_product([line,Seg])
    Cong_sg_data=pd.DataFrame(index=seg_ind)


    
    # ds_loading=0.8 #desired loading
    #gamm=4 gamma*2 is the power
    for l in line:
        n=1
        for sg_order,sg in enumerate(Seg):
            
            if sg_order==0: #first segment
                Cong_sg_data.loc[(l,sg),'df']=((ds_loading)*limits[line.index(l)]   )
                Cong_sg_data.loc[(l,sg),'fini']=(n-1)*Cong_sg_data.loc[(l,sg),'df']
                Cong_sg_data.loc[(l,sg),'ffin']=Cong_sg_data.loc[(l,sg),'fini']+Cong_sg_data.loc[(l,sg),'df']
                Cong_sg_data.loc[(l,sg),'Cini']=(Cong_sg_data.loc[(l,sg),'fini']/limits[line.index(l)])**(2*gamma)
                Cong_sg_data.loc[(l,sg),'Cfin']=(Cong_sg_data.loc[(l,sg),'ffin']/limits[line.index(l)])**(2*gamma)
                Cong_sg_data.loc[(l,sg),'s']=0.0
                n+=1
            
            if sg_order!=0:
                Cong_sg_data.loc[(l,sg),'df']=((1-ds_loading)*limits[line.index(l)] )/(len(Seg)-1)
                Cong_sg_data.loc[(l,sg),'fini']=((ds_loading)*limits[line.index(l)] ) + \
                                                                (n-2)*Cong_sg_data.loc[(l,sg),'df']
                Cong_sg_data.loc[(l,sg),'ffin']=Cong_sg_data.loc[(l,sg),'fini']+Cong_sg_data.loc[(l,sg),'df']
                Cong_sg_data.loc[(l,sg),'Cini']=(Cong_sg_data.loc[(l,sg),'fini']/limits[line.index(l)])**(2*gamma)
                Cong_sg_data.loc[(l,sg),'Cfin']=(Cong_sg_data.loc[(l,sg),'ffin']/limits[line.index(l)])**(2*gamma)
                Cong_sg_data.loc[(l,sg),'s']=(Cong_sg_data.loc[(l,sg),'Cfin']-Cong_sg_data.loc[(l,sg),'Cini'])/Cong_sg_data.loc[(l,sg),'df']
                n+=1
            
            if sg_order == len(Seg)-1: #last segment

                Cong_sg_data.loc[(l,sg),'df'] = limits[line.index(l)] # we allow ~100% overloading with this

                #increase the df of the last sg so that it's not limited to max flow limit
    
    return Cong_sg_data, Seg





def Build_graph_default(nodes,Lines):

    node_list=nodes.copy()

    graph=nx.Graph()

    graph.add_nodes_from(node_list)
    for l,i,j in Lines:
        graph.add_edge(i, j)

    return graph




# %%
#====================================================================== 
def Find_radial_lines(Lines,nodes):
    
    node_list=nodes.copy()
    
    G=nx.Graph()
    
    edge_list=[(i,j) for l,i,j in Lines]
    
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    
    radial_lines=[]
    for l,i,j in Lines:
        G.remove_edge(i,j)
        if G.degree(i) == 0 or G.degree(j) == 0:
            if l not in radial_lines:
                radial_lines.append(l)
        G.add_edge(i,j)

    return radial_lines



def find_non_radial_Nk(Lines, nodes, k=1):
    """
    Returns all combinations of k lines whose removal does NOT isolate any node.
    
    Args:
        Lines: list of (l, i, j)
        nodes: list of node IDs
        k: number of line outages to consider

    Returns:
        valid_combos: list of outage line combinations (each as list of line IDs)
    """
    edge_list = [(i, j, l) for l, i, j in Lines]
    valid_combos = []

    for combo in itertools.combinations(Lines, k):
        outage_ids = {l for l, i, j in combo}

        # Build graph without the outage lines
        G_temp = nx.Graph()
        G_temp.add_nodes_from(nodes)
        for l, i, j in Lines:
            if l not in outage_ids:
                G_temp.add_edge(i, j)

        # Check if any node got isolated
        isolated = any(G_temp.degree(n) == 0 for n in nodes)
        
        if not isolated:
            valid_combos.append(list(outage_ids))

    return valid_combos



# %%


# %%
def find_k_hop_nodes_from_congestion(Bus,Lines1D, overloaded_mask,lines_outage=[], k=5,min_degree=4):
    """
    Finds all nodes that are within k hops from any congested line.
    
    Args:
        Lines1D: list of (l, i, j)
        overloaded_mask: boolean list or array for congested lines (same length as Lines1D)
        k: number of hops
        min_degree: minimum number of line connections a node must have to be included

    Returns:
        affected_nodes: set of node indices within k hops from congestion
    """
    # Step 1: Build graph
    G = nx.Graph()
    for l, i, j in Lines1D:
        if l not in lines_outage:
            G.add_edge(i, j)
    
    node_degrees = [G.degree[i] for i in Bus]

    # Step 2: Identify congested edges and their endpoint nodes
    congested_nodes = set()
    for (is_overloaded, (line_id, i, j)) in zip(overloaded_mask, Lines1D):
        if is_overloaded and line_id not in lines_outage:
            congested_nodes.update([i, j])

    # Step 3: BFS from each congested node, gather nodes within k hops
    affected_nodes = set()
    for node in congested_nodes:
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=k)
        affected_nodes.update(lengths.keys())

    high_degree_nodes = [n for n in affected_nodes if G.degree[n] >= min_degree]

    return sorted(high_degree_nodes), node_degrees



def find_k_hop_nodes_from_split(Bus,Lines1D,split_node=[],lines_outage=[], k=10):
    """
    Finds all nodes that are within k hops from any node.
    
    Args:
        Lines1D: list of (l, i, j)
       split_node: the splitting node to find k-hops from
        k: number of hops

    Returns:
        affected_nodes: set of node indices within k hops from split_node
    """
    # Step 1: Build graph
    G = nx.Graph()
    for l, i, j in Lines1D:
        if l not in lines_outage:
            G.add_edge(i, j)
    
    node_degrees = [G.degree[i] for i in Bus]


    # Step 2: BFS from each split node and gather nodes within k hops
    affected_nodes = set()
    for node in split_node:
        if node in G:  # Avoid errors if node is isolated due to outages
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=k)
            affected_nodes.update(lengths.keys())
    
    affected_nodes = list(affected_nodes)
    not_affected_nodes = [b for b in Bus if b not in affected_nodes ]

    return affected_nodes, not_affected_nodes



# %%
def find_radial_end_nodes(Lines1D):
    """
    Identifies nodes connected to dead-end (radial) branches.
        
    Returns:
        List of dead-end nodes (degree 1)
    """
    G = nx.Graph()

    # Add edges (ignore line_id)
    for _, i, j in Lines1D:
        G.add_edge(i, j)

    visited = set()
    all_chain_nodes = set()  # Use a set to avoid duplicates

    for node in G.nodes():
        if G.degree(node) == 1 and node not in visited:
            current_chain = [node]
            current = node
            prev = None

            while True:
                neighbors = [n for n in G.neighbors(current) if n != prev]
                if not neighbors:
                    break
                next_node = neighbors[0]
                current_chain.append(next_node)

                if G.degree(next_node) > 2 or next_node in visited:
                    break

                prev, current = current, next_node

            visited.update(current_chain)
            all_chain_nodes.update(current_chain)

    return list(all_chain_nodes)


# %%

def node_distance_from_congestion(Bus, Lines1D, overloaded_mask, lines_outage=[]):
    """
    Computes the number of hops from each node to the nearest congested line.

    Args:
        Bus: list of all bus indices
        Lines1D: list of tuples (line_id, bus_i, bus_j)
        overloaded_mask: boolean list or array for congested lines (same length as Lines1D)
        lines_outage: list of line IDs that are out of service (to exclude from graph)

    Returns:
        hop_distances: list {hops_to_nearest_congested_line}
                      nodes unreachable from congestion will have value -1
    """
    # Step 1: Build graph
    G = nx.Graph()
    for l, i, j in Lines1D:
        if l not in lines_outage:
            G.add_edge(i, j)

    # Step 2: Identify endpoint nodes of congested lines
    congested_nodes = set()
    for (is_overloaded, (line_id, i, j)) in zip(overloaded_mask, Lines1D):
        if is_overloaded and line_id not in lines_outage:
            congested_nodes.update([i, j])

    if not congested_nodes:
        # No congestion: all nodes get -1
        return [+1000 for node in Bus]

    # Step 3: Multi-source BFS to compute shortest paths
    lengths = nx.multi_source_dijkstra_path_length(G, sources=congested_nodes, weight=None)

    # Step 4: Assign distances to each node, +1K if not reachable
    hop_distances = [ lengths.get(node, +1000) for node in Bus]

    return hop_distances



# %%
def Create_DC_OPF_Pyomo(data):
    """
    Create a DC optimal power flow model in Pyomo.

    The model minimizes generation cost subject to DC power-flow equations,
    nodal active-power balance, generator limits, line-flow limits, and optional
    line outages. The outage set is mutable so that different contingency cases
    can be evaluated without rebuilding the model from scratch.

    Parameters
    ----------
    data : dict
        Input dictionary produced by `read_data_AC`, containing network sets,
        branch parameters, demand data, generation data, and congestion segments.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        Pyomo DC-OPF model.
    """

    # ======  data
    Bus = data['Bus']
    branch = data['branch']
    Lines = data['Lines']
    line = data['line']
    Lines1D = data['Lines1D']
    Gen_data = data['Gen_data']
    G = data['G']
    G2B = data['G2B']
    Seg = data['Seg']


    model = pyo.ConcreteModel()

    # === Sets
    model.G = pyo.Set(initialize=G)
    model.Bus = pyo.Set(initialize=Bus)
    model.Lines = pyo.Set(initialize=Lines, dimen=3)
    model.Lines1D = pyo.Set(initialize=Lines1D, dimen=3)
    model.Seg = pyo.Set(initialize=Seg)
    model.LineSeg = pyo.Set(initialize=[(l, sg) for l in line for sg in Seg], dimen=2)

    # Outage line set (mutable so you can update it later)
    model.outage_lines = pyo.Set(initialize=[], dimen=1)

    # === Params

    # Demand parameters
    Pd_init = {d: data['Pdemand'].loc[d, 'Pd'] for d in data['Demandset']}
    model.DemandSet = pyo.Set(initialize=data['Demandset'])
    model.Pd = pyo.Param(model.DemandSet, initialize=Pd_init, mutable=True)

    # Line-limit parameters
    limit_init = {(l, i, j): branch.loc[(l, i, j), 'limit'] for l, i, j in Lines}
    model.limit = pyo.Param(model.Lines, initialize=limit_init, mutable=True)

    # Generator-cost parameters
    gen_cost_init = {g: Gen_data.loc[g, 'b'] for g in G}
    model.gen_cost_b = pyo.Param(model.G, initialize=gen_cost_init, mutable=True)
    
    # === Variables
    # Generation variables
    model.Pg = pyo.Var(model.G, domain=pyo.NonNegativeReals)

    # Voltage-angle variables
    model.delta = pyo.Var(model.Bus)

    # Line-flow variables
    model.Pflow = pyo.Var(model.Lines)

    # Positive and negative flow variables for piecewise congestion modeling
    model.Pflow_ps = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.Pflow_ng = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)

    # Congestion-cost variables
    model.CongCost = pyo.Var(line, domain=pyo.NonNegativeReals)
    model.TotalCongCost = pyo.Var(domain=pyo.NonNegativeReals)

    # === Constraints
    # Flow constraint with line outage skip
    def flow_eq_with_outage(m, l, i, j):
        """Enforce DC line-flow equation for lines that are in service."""

        if l in m.outage_lines:
            return pyo.Constraint.Skip

        return m.Pflow[l, i, j] == (m.delta[i] - m.delta[j]) / data['branch'].loc[(l, i, j), 'x']

    model.flow_eq = pyo.Constraint(model.Lines, rule=flow_eq_with_outage)


    def zero_flow_eq(m, l, i, j):
        """Force line flow to zero for outage lines."""

        if l in m.outage_lines:
            return m.Pflow[l, i, j] == 0

        return pyo.Constraint.Skip  # do not add anything for lines in service

    model.zero_flow_eq = pyo.Constraint(model.Lines, rule=zero_flow_eq)


    def balance_eq(m, b):
        """Enforce active-power balance at each bus."""

        gen_sum = sum(m.Pg[g] for g, bb in data['G2B'] if bb == b)
        demand_sum = sum(m.Pd[d] for d, bb in data['D2B'] if bb == b)
        inflow = sum(m.Pflow[l, i, j] for l, i, j in data['Lines'] if i == b)

        return gen_sum - demand_sum == inflow

    model.balance_eq = pyo.Constraint(model.Bus, rule=balance_eq)


    # Line-flow limits
    model.pij_max = pyo.Constraint(
        model.Lines,
        rule=lambda m, l, i, j: m.Pflow[l, i, j] <= m.limit[l, i, j],
    )

    model.pij_min = pyo.Constraint(
        model.Lines,
        rule=lambda m, l, i, j: m.Pflow[l, i, j] >= -m.limit[l, i, j],
    )

    # Generator operating limits
    model.pg_max = pyo.Constraint(
        model.G,
        rule=lambda m, g: m.Pg[g] <= Gen_data.loc[g, 'Pmax'],
    )

    model.pg_min = pyo.Constraint(
        model.G,
        rule=lambda m, g: m.Pg[g] >= Gen_data.loc[g, 'Pmin'],
    )


    # Reference angle constraint
    model.delta_ref = pyo.Constraint(expr=model.delta[Bus[0]] == 0)


    def quad_cost_rule(model):
        """Compute the quadratic generation-cost objective."""

        return sum(
            Gen_data.loc[g]['a'] * model.Pg[g] ** 2
            + model.gen_cost_b[g] * model.Pg[g]
            for g in model.G
        )

    # === Objective
    model.obj = pyo.Objective(rule=quad_cost_rule, sense=pyo.minimize)

    return model     
    
    

# %%
def Solve_DC_OPF_Pyomo(
    data,
    modelX,
    Pd_instance=None,
    lines_outage=[],
    limit_new=None,
    gen_cost_new=None,
    cong_limit=0.95,
    print_result=False,
):
    """
    Solve a DC optimal power flow model for a given demand, outage, limit, and cost scenario.

    The function clones a pre-built Pyomo DC-OPF model, updates mutable parameters
    such as demand, line limits, generator costs, and outage lines, solves the model
    with Gurobi, and returns generation dispatch, line flows, nodal injections,
    voltage angles, congestion indicators, and execution time.

    Parameters
    ----------
    data : dict
        Input data dictionary containing network sets, demand data, generation data,
        branch data, and connectivity mappings.
    modelX : pyomo.environ.ConcreteModel
        Base DC-OPF Pyomo model to be cloned and solved.
    Pd_instance : array-like, optional
        Active demand values for all demand points.
    lines_outage : list
        List of line IDs that are out of service.
    limit_new : array-like, optional
        Updated line limits for one-directional lines.
    gen_cost_new : array-like, optional
        Updated linear generation-cost coefficients.
    cong_limit : float
        Loading-ratio threshold used to identify congested lines.
    print_result : bool
        If True, print generation cost and number of congested lines.

    Returns
    -------
    dict or str
        Dictionary of OPF results if the model is feasible and solved optimally;
        otherwise, returns "Infeasible".
    """

    # === Network data
    Bus = data['Bus']
    G = data['G']
    branch = data['branch']
    Lines = data['Lines']
    line = data['line']
    Lines1D = data['Lines1D']
    G2B = data['G2B']
    D2B = data['D2B']

    # Clone the base model so that the original model remains unchanged.
    model = modelX.clone()

    if Pd_instance is None:
        print("DemandInstance is required")
        Pd_instance = data['Pdemand']['Pd'].to_numpy()

    # === Update Pd param
    for d_ind, d in enumerate(data['Demandset']):
        model.Pd[d] = Pd_instance[d_ind]

    # === Update line-limit parameters
    if limit_new is not None:
        for l_ind, (l, i, j) in enumerate(Lines1D):
            model.limit[l, i, j] = limit_new[l_ind]
            model.limit[l, j, i] = limit_new[l_ind]   # bi-directional lines

    # === Update generator-cost parameters
    if gen_cost_new is not None:
        for g_ind, g in enumerate(G):
            model.gen_cost_b[g] = gen_cost_new[g_ind]

    # === Update outage set and reconstruct flow constraint
    model.outage_lines.clear()
    model.outage_lines.update(lines_outage)

    model.flow_eq.clear()
    model.flow_eq._constructed = False
    model.flow_eq.construct()

    model.zero_flow_eq.clear()
    model.zero_flow_eq._constructed = False
    model.zero_flow_eq.construct()

    # model.write('dcopf_debug.lp', io_options={'symbolic_solver_labels': True})

    # === Solve
    try:
        solver = pyo.SolverFactory("gurobi")

        start_time = time.time()
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)

        result = SolverFactory("gurobi").solve(
            model,
            tee=False,
            options={"OutputFlag": 0},
        )

        end_time = time.time()
        ex_time = end_time - start_time

        # === Check feasibility
        if (
            result.solver.termination_condition != pyo.TerminationCondition.optimal
            or result.solver.status != SolverStatus.ok
        ):
            return "Infeasible"

        # === Flow values
        if limit_new is not None:
            limit = limit_new.copy()
        else:
            limit = data['branch']['limit'].to_numpy()[:len(data['line'])]

        flows = np.zeros(len(line))
        Pflow_df = pd.DataFrame(columns=['flow'], index=data['branch1D'].index)

        for l_ind, (l, i, j) in enumerate(Lines1D):
            flows[l_ind] = model.Pflow[l, i, j].value
            Pflow_df.loc[(l, i, j), 'flow'] = model.Pflow[l, i, j].value

        # === Congestion indicators
        loading_ratio = flows / limit
        overloaded_mask = np.abs(loading_ratio) >= cong_limit
        Num_cong = np.sum(overloaded_mask)

        # === Flows on lines in service
        Lines_in_service = [(l, i, j) for l, i, j in Lines1D if l not in lines_outage]
        Pflow_top = np.zeros(len(Lines_in_service))

        for idx, (l, i, j) in enumerate(Lines_in_service):
            Pflow_top[idx] = model.Pflow[l, i, j].value

        # === Pg values
        Pgdict = {}
        Pg_array = np.zeros(len(G))

        for g_ind, g in enumerate(G):
            Pg_val = model.Pg[g].value
            Pgdict[g] = Pg_val
            Pg_array[g_ind] = Pg_val

        # === Nodal injections
        Pinj = np.zeros((len(Bus)))

        for b_ind, b in enumerate(Bus):
            for g, bb in G2B.select('*', b):
                Pinj[b_ind] += model.Pg[g].value

            for d, bb in D2B.select('*', b):
                Pinj[b_ind] -= model.Pd[d].value

        # === delta values
        delta = np.zeros(len(Bus))

        for b_ind, b in enumerate(Bus):
            delta[b_ind] = model.delta[b].value

        # === Print if needed
        if print_result:
            print(f"\t GenCost: {pyo.value(model.obj):.3f}")
            print(f"\t Num of congested lines: {Num_cong}")

        return {
            'GenCost': pyo.value(model.obj),
            'time': ex_time,
            'Pg': Pgdict,
            'Pg_array': Pg_array,
            'Pflow': flows,
            'Pflow_top': Pflow_top,
            'Pflow_df': Pflow_df,
            'Pinj': Pinj,
            'delta': delta,
            'Num_cong': Num_cong,
            'overloaded_mask': overloaded_mask,
        }

    except:
        return "Infeasible"



    

# %% Example
# data = read_data_AC('IEEE_14_bus_Data_PGLib_ACOPF.xlsx',DemFactor=4.0,LineLimit=1.0,Seg_num=5,gamma=10)
# Pd = data['Pdemand']['Pd'].to_numpy()
# limits = data['branch1D']['limit'].to_numpy()
# gcost = 2*data['Gen_data']['b'].to_numpy()
# modelX = Create_DC_OPF_Pyomo(data)
# res = Solve_DC_OPF_Pyomo(data=data,modelX=modelX,Pd_instance=Pd,lines_outage=[], limit_new= limits, gen_cost_new=gcost ,print_result=True)



# %% 
# OSS Optimal Substation Switching 

def Create_Cong_DC_OSS_Pyomo(
    data,
    Substation_nodes=[],
    Max_Sw_bus=0,
    LineLimit=1.0,   # you can relax the line limit to allow switching
    Zinitial=None,
    Zfixdict=None,
):
    """
    Create a congestion-management DC optimal substation switching (OSS) Pyomo model.

    This model performs real-time topology optimization using busbar splitting
    while keeping generator dispatch fixed from a previous DC-OPF solution.
    The formulation minimizes congestion cost subject to DC power-flow equations,
    busbar-splitting constraints, line-flow limits, and topology-switching logic.

    Parameters
    ----------
    data : dict
        Input network and system data dictionary.
    Substation_nodes : list
        List of buses modeled as substations with switchable busbars.
        If empty, all buses are modeled as substations.
    Max_Sw_bus : int
        Maximum number of substations allowed to split.
    LineLimit : float
        Scaling factor applied to line limits.
    Zinitial : dict, optional
        Initial binary values for warm-starting switching variables.
    Zfixdict : dict, optional
        Dictionary for fixing switching configurations.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        Pyomo congestion-management OSS model.
    """
    
    #======  data
    
    Sbase=data['Sbase']
    Max_MIPGap=data['Max_MIPGap']
    Max_timelimit=data['Max_timelimit'] #=300 #900
    Bus=data['Bus']    # for b in Bus
    busbar=data['busbar']
    branch=data['branch']
    Lines=data['Lines']
    line=data['line']
    Lines1D=data['Lines1D']
    L2B=data['L2B']
    NumberL2B=data['NumberL2B']
    DemandSet=data['Demandset']     
    D2B=data['D2B']                 
    Gen_data=data['Gen_data']
    G=data['G']            
    G2B=data['G2B']            

    Seg=data['Seg']
    Cong_sg_data=data['Cong_sg_data']
    
    #======
    BigM_l=data['BigM_l'] 
    BigM_b=data['BigM_b']
    Maxdelta=data['Maxdelta'] 

    if Substation_nodes==[]:
        print('All nodes are modeled as substations.')
        Bus_sub = Bus.copy()
    else:
        Bus_sub = Substation_nodes.copy()
    
    
    model = pyo.ConcreteModel()

    # === Sets
    model.Bus = pyo.Set(initialize=Bus)
    model.busbar = pyo.Set(initialize=busbar)
    model.Lines = pyo.Set(initialize=Lines, dimen=3)
    model.Lines1D = pyo.Set(initialize=Lines1D, dimen=3)
    model.line = pyo.Set(initialize=line)
    model.G = pyo.Set(initialize=G)
    model.DemandSet = pyo.Set(initialize=DemandSet)
    model.Seg = pyo.Set(initialize=Seg)

    # === Substation definitions
    Bus_sub = Substation_nodes if Substation_nodes else Bus
    Bus_non_sub = [b for b in Bus if b not in Bus_sub]

    G_sub = [g for sub in Bus_sub for g, x in G2B.select('*', sub)]
    G_non_sub = [g for g in G if g not in G_sub]

    D_sub = [d for sub in Bus_sub for d, x in D2B.select('*', sub)]
    D_non_sub = [d for d in DemandSet if d not in D_sub]

    Lines_sub = [(l, i, j) for sub in Bus_sub for l, i, j in Lines.select('*', sub, '*')]

    model.Bus_sub = pyo.Set(initialize=Bus_sub)
    model.Bus_non_sub = pyo.Set(initialize=Bus_non_sub)
    model.G_sub = pyo.Set(initialize=G_sub)
    model.G_non_sub = pyo.Set(initialize=G_non_sub)
    model.D_sub = pyo.Set(initialize=D_sub)
    model.D_non_sub = pyo.Set(initialize=D_non_sub)
    model.Lines_sub = pyo.Set(initialize=Lines_sub, dimen=3)
    model.LineSeg = pyo.Set(initialize=[(l, sg) for l in line for sg in Seg], dimen=2)

    # Outage line set (mutable so you can update it later)
    model.outage_lines = pyo.Set(initialize=[],dimen=1)

    # === Params
    Pd_init = {d: data['Pdemand'].loc[d, 'Pd'] for d in data['Demandset']}
    model.Pd_param = pyo.Param(model.DemandSet, initialize=Pd_init, mutable=True)

    Pg_init = {g: 1 for g in G}
    model.Pg_DCOPF = pyo.Param(model.G, initialize=Pg_init,mutable=True)

    limit_init = {(l,i,j): branch.loc[(l, i, j), 'limit'] for l, i, j in Lines}
    model.limit = pyo.Param(model.Lines, initialize=limit_init, mutable=True)

    gen_cost_init = {g: Gen_data.loc[g, 'b'] for g in G}
    model.gen_cost_b = pyo.Param(model.G, initialize=gen_cost_init, mutable=True)

    cong_cost_init = {(l,sg): Cong_sg_data.loc[(l, sg), 's']  for l,sg in model.LineSeg  }
    model.cong_cost_s = pyo.Param(model.LineSeg, initialize=cong_cost_init, mutable=True)

    cong_df_init = {(l,sg): Cong_sg_data.loc[(l, sg), 'df']  for l,sg in model.LineSeg}
    model.cong_df = pyo.Param(model.LineSeg, initialize=cong_df_init, mutable=True)

    

    # === Variables
    model.Pg = pyo.Var(model.G, domain=pyo.NonNegativeReals)
    model.Pgi = pyo.Var(model.G_sub, model.busbar, domain=pyo.NonNegativeReals)

    model.Pd = pyo.Var(D_non_sub, domain=pyo.NonNegativeReals)
    model.Pdi = pyo.Var(model.D_sub, model.busbar, domain=pyo.NonNegativeReals)

    model.Pflow = pyo.Var(model.Lines)
    model.Pflow_li = pyo.Var(model.Lines_sub, model.busbar)

    model.delta_bi = pyo.Var(model.Bus_sub, model.busbar)
    model.delta_li = pyo.Var(model.Lines)

    model.z_bus = pyo.Var(model.Bus_sub, domain=pyo.Binary)
    model.z_li = pyo.Var(model.Lines_sub, domain=pyo.Binary)
    model.z_g = pyo.Var(model.G_sub, domain=pyo.Binary)
    model.z_d = pyo.Var(model.D_sub, domain=pyo.Binary)

    model.Pflow_ps = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.Pflow_ng = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.CongCost = pyo.Var(model.line, domain=pyo.NonNegativeReals)
    model.TotalCongCost = pyo.Var(domain=pyo.NonNegativeReals)
    # model.GenCost = pyo.Var(domain=pyo.NonNegativeReals)
    model.OF = pyo.Var(domain=pyo.NonNegativeReals)


    # Initial values
    if Zinitial is None:
        for b in Bus_sub:
            model.z_bus[b].value = 1
        for l, i, j in Lines_sub:
            model.z_li[l, i, j].value = 0
        for g in G_sub:
            model.z_g[g].value = 0
        for d in D_sub:
            model.z_d[d].value = 0
    else:
        for b in Bus_sub:
            model.z_bus[b].value = Zinitial['bus'][b]
        for l, i, j in Lines_sub:
            model.z_li[l, i, j].value = Zinitial['l_i'][l, i, j]
        for g in G_sub:
            model.z_g[g].value = Zinitial['g'][g]
        for d in D_sub:
            model.z_d[d].value = Zinitial['d'][d]



    # Constraints for 2-line rule
    def eq_2line_busbar2_rule(m, b):
        return 2 * (1 - m.z_bus[b]) <= sum(m.z_li[l, i, j] for (l, i, j) in Lines.select('*',b,'*'))
    model.eq_2line_busbar2 = Constraint(model.Bus_sub, rule=eq_2line_busbar2_rule)

    def eq_2line_busbar1_rule(m, b):
        return 2 * (1 - m.z_bus[b]) <= sum(1 - m.z_li[l, i, j] for (l, i, j) in Lines.select('*',b,'*'))
    model.eq_2line_busbar1 = Constraint(model.Bus_sub, rule=eq_2line_busbar1_rule)

    def eq_2line_busbar3_rule(m, b):
        if NumberL2B[b] <= 3:
            return m.z_bus[b] == 1
        return Constraint.Skip
    model.eq_2line_busbar3 = Constraint(model.Bus_sub, rule=eq_2line_busbar3_rule)

    model.eq_MaxSw_bus = Constraint(expr=sum(1 - model.z_bus[b] for b in Bus_sub) <= Max_Sw_bus)

    #    symmetry 
    if Zfixdict==None:
        model.eq_symmetry = pyo.ConstraintList()
        for b in Bus_sub:
            lmin,b=L2B.select('*',b)[0]
            lmin,i,j=Lines.select(lmin,b,'*')[0]
            model.eq_symmetry.add(model.z_li[lmin, i, j] == 0)

    # Tightening bounds
    model.eq_bg = ConstraintList()
    for b in Bus_sub:
        for g, bb in G2B.select('*',b):
            model.eq_bg.add(model.z_bus[b] - 1 + model.z_g[g] <= 0)

    model.eq_bd = ConstraintList()
    for b in Bus_sub:
        for d, bb in D2B.select('*',b):
            model.eq_bd.add(model.z_bus[b] - 1 + model.z_d[d] <= 0)

    model.eq_blFe = ConstraintList()
    for b in Bus_sub:
        for l, i, j in Lines.select('*',b,'*'):
            model.eq_blFe.add(model.z_bus[b] - 1 + model.z_li[l, i, j] <= 0)


    def eq_Pg_rule(m, g):
        return m.Pg[g] == m.Pgi[g, 'busbar1'] + m.Pgi[g, 'busbar2']
    model.eq_Pg = pyo.Constraint(model.G_sub, rule=eq_Pg_rule)

    
    
    # For generators in substations (split between busbar1 and busbar2)
    model.eq_Pg1 = pyo.Constraint(model.G_sub, rule=lambda m, g: 
        m.Pgi[g, 'busbar1'] == (1 - m.z_g[g]) * m.Pg_DCOPF[g])

    model.eq_Pg2 = pyo.Constraint(model.G_sub, rule=lambda m, g: 
        m.Pgi[g, 'busbar2'] == m.z_g[g] * m.Pg_DCOPF[g])

    # For generators not in substations (single busbar)
    model.eq_Pg_non_sub = pyo.Constraint(model.G_non_sub, rule=lambda m, g: 
        m.Pg[g] == m.Pg_DCOPF[g])
    

    #Dem
    model.eq_Pd1 = Constraint(model.D_sub, rule=lambda m, d: 
        m.Pdi[d, 'busbar1'] == (1 - m.z_d[d]) * m.Pd_param[d])

    model.eq_Pd2 = Constraint(model.D_sub, rule=lambda m, d: 
        m.Pdi[d, 'busbar2'] == m.z_d[d] * m.Pd_param[d])
    
    model.eq_Pd = Constraint(model.D_non_sub, rule=lambda m, d: 
        m.Pd[d] == m.Pd_param[d])
    

    
    # DC Equations
    def flow_eq_with_outage(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        return m.Pflow[l, i, j] == (m.delta_li[l,i,j] - m.delta_li[l,j,i]) / data['branch'].loc[(l, i, j), 'x']
    model.flow_eq = pyo.Constraint(model.Lines, rule=flow_eq_with_outage)

    def zero_flow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Pflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_flow_eq = pyo.Constraint(model.Lines, rule=zero_flow_eq)

    
    model.pij_max = pyo.Constraint(model.Lines, rule=lambda m, l, i, j:
        m.Pflow[l, i, j] <= m.limit[l,i,j]*LineLimit )

    model.pij_min = pyo.Constraint(model.Lines, rule=lambda m, l, i, j:
        -m.limit[l,i,j]*LineLimit <= m.Pflow[l, i, j] )



    def eq_flow1min_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit <= m.Pflow_li[l, i, j, 'busbar1']
    model.eq_flow1min = pyo.Constraint(model.Lines_sub, rule=eq_flow1min_rule)

    def eq_flow1max_rule(m, l, i, j):
        return m.Pflow_li[l, i, j, 'busbar1'] <= (1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit
    model.eq_flow1max = pyo.Constraint(model.Lines_sub, rule=eq_flow1max_rule)

    def eq_flow2min_rule(m, l, i, j):
        return -m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit <= m.Pflow_li[l, i, j, 'busbar2']
    model.eq_flow2min = pyo.Constraint(model.Lines_sub, rule=eq_flow2min_rule)

    def eq_flow2max_rule(m, l, i, j):
        return m.Pflow_li[l, i, j, 'busbar2'] <= m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit
    model.eq_flow2max = pyo.Constraint(model.Lines_sub, rule=eq_flow2max_rule)

    def eq_flow_sum_rule(m, l, i, j):
        return m.Pflow[l, i, j] == m.Pflow_li[l, i, j, 'busbar1'] + m.Pflow_li[l, i, j, 'busbar2']
    model.eq_Pflow = pyo.Constraint(model.Lines_sub, rule=eq_flow_sum_rule)



    #delta
    def eq_delta_bus1_rule(m, b):
        return -BigM_b * (1 - m.z_bus[b]) <= m.delta_bi[b, 'busbar1'] - m.delta_bi[b, 'busbar2']
    model.eq_delta_bus1 = pyo.Constraint(model.Bus_sub, rule=eq_delta_bus1_rule)

    def eq_delta_bus2_rule(m, b):
        return m.delta_bi[b, 'busbar1'] - m.delta_bi[b, 'busbar2'] <= BigM_b * (1 - m.z_bus[b])
    model.eq_delta_bus2 = pyo.Constraint(model.Bus_sub, rule=eq_delta_bus2_rule)

    # Delta difference bounds for lines
    def eq_delta_lb1Frmin_rule(m, l, i, j):
        return -m.z_li[l, i, j] * Maxdelta <= m.delta_li[l, i, j] - m.delta_bi[i, 'busbar1']
    model.eq_delta_lb1Frmin = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb1Frmin_rule)

    def eq_delta_lb1Frmax_rule(m, l, i, j):
        return m.delta_li[l, i, j] - m.delta_bi[i, 'busbar1'] <= m.z_li[l, i, j] * Maxdelta
    model.eq_delta_lb1Frmax = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb1Frmax_rule)

    def eq_delta_lb2Frmin_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * Maxdelta <= m.delta_li[l, i, j] - m.delta_bi[i, 'busbar2']
    model.eq_delta_lb2Frmin = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb2Frmin_rule)

    def eq_delta_lb2Frmax_rule(m, l, i, j):
        return m.delta_li[l, i, j] - m.delta_bi[i, 'busbar2'] <= (1 - m.z_li[l, i, j]) * Maxdelta
    model.eq_delta_lb2Frmax = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb2Frmax_rule)

    #delta i
    model.eqDelta = pyo.ConstraintList()
    for b in Bus_non_sub:
        lmin, b, j1 = Lines.select('*', b, '*')[0] #first line
        for l, i, j in Lines.select('*', b, '*'):
            model.eqDelta.add(
                    model.delta_li[lmin, b, j1] == model.delta_li[l, i, j])
    
    # Reference angle constraint
    l0, i0, j0 = Lines[0]
    model.ref_bus_angle = pyo.Constraint(expr=model.delta_li[l0, i0, j0] == 0)



    # === Balance constraint for substation buses
    def eq_balance_sub_rule(m, b, mbar):
        gen_sum = sum(m.Pgi[g, mbar] for g, bb in G2B.select('*',b))
        load_sum = sum(m.Pdi[d, mbar] for d, bb in D2B.select('*',b))
        flow_sum = sum(m.Pflow_li[l, i, j, mbar] for l, i, j in Lines.select('*',b,'*'))
        return gen_sum - load_sum == flow_sum
    model.eq_balance_sub = pyo.Constraint(model.Bus_sub, model.busbar, rule=eq_balance_sub_rule)

    # === Balance constraint for non-substation buses
    def eq_balance_nonsub_rule(m, b):
        gen_sum = sum(m.Pg[g] for g, bb in G2B.select('*',b))
        load_sum = sum(m.Pd[d] for d, bb in D2B.select('*',b))
        flow_sum = sum(m.Pflow[l, i, j] for l, i, j in Lines.select('*',b,'*'))
        return gen_sum - load_sum == flow_sum
    model.eq_balance_nonsub = pyo.Constraint(model.Bus_non_sub, rule=eq_balance_nonsub_rule)


    # === Congestion power flow decomposition
    def eqPflowSumCong_rule(m, l, i, j):
        return m.Pflow[l, i, j] == sum(m.Pflow_ps[l, sg] for sg in m.Seg) - sum(m.Pflow_ng[l, sg] for sg in m.Seg)
    model.eqPflowSumCong = pyo.Constraint(model.Lines1D, rule=eqPflowSumCong_rule)

    # === Upper bounds on Pflow_ps
    def eqPflow_ps_SgMax_rule(m, l, sg):
        return m.Pflow_ps[l, sg] <= model.cong_df[l,sg]
    model.eqPflow_ps_SgMax = pyo.Constraint(model.line, model.Seg, rule=eqPflow_ps_SgMax_rule)

    # === Upper bounds on Pflow_ng
    def eqPflow_ng_SgMax_rule(m, l, sg):
        return m.Pflow_ng[l, sg] <= model.cong_df[l,sg]  
    model.eqPflow_ng_SgMax = pyo.Constraint(model.line, model.Seg, rule=eqPflow_ng_SgMax_rule)

    # === Congestion cost per line
    def eqCongCost_rule(m, l, i, j):
        return m.CongCost[l] == sum(m.Pflow_ps[l, sg] * m.cong_cost_s[l,sg] for sg in m.Seg) + \
                                sum(m.Pflow_ng[l, sg] * m.cong_cost_s[l,sg] for sg in m.Seg)
    model.eqCongCost = pyo.Constraint(model.Lines1D, rule=eqCongCost_rule)

    # === Total congestion cost
    model.eqTotalCongCost = pyo.Constraint(expr=model.TotalCongCost == sum(model.CongCost[l] for l in model.line))

    # === Link total cost to objective
    model.Eq_OF = pyo.Constraint(expr=model.OF == model.TotalCongCost)

    # === Objective function
    model.obj = pyo.Objective(expr=model.OF, sense=pyo.minimize)


    return model






# %%
def Solve_Cong_DC_OSS_Pyomo(
    data,
    modelX,
    Substation_nodes=[],
    lines_outage=[],
    res_DCOPF=None,
    Pd_instance=None,
    limit_new=None,
    gen_cost_new=None,
    candidate_bus=None,
    force_split=[],
    cong_limit=0.95,
    print_result=False,
):
    """
    Solve the congestion-management DC optimal substation switching model.

    'RealTime': the dispatch of generators is not changed. we find a switching that can minimize congestion.

    The function clones a pre-built OSS model, updates demand, fixed generator
    dispatch, line limits, outage lines, and optional candidate/forced switching
    decisions. It then solves the model and returns line flows, congestion status,
    opened busbar couplers, and execution time.

    Parameters
    ----------
    data : dict
        Input network and system data dictionary.
    modelX : pyomo.environ.ConcreteModel
        Base congestion-management DC-OSS Pyomo model to be cloned and solved.
    Substation_nodes : list
        List of buses modeled as switchable substations.
    lines_outage : list
        List of line IDs that are out of service.
    res_DCOPF : dict, optional
        DC-OPF result dictionary containing fixed generator dispatch.
    Pd_instance : array-like, optional
        Active demand values for all demand points.
    limit_new : array-like, optional
        Updated line limits for one-directional lines.
    gen_cost_new : array-like, optional
        Updated generator-cost coefficients.
    candidate_bus : list, optional
        Candidate substations allowed to split. Non-candidate substations are fixed closed.
    force_split : list
        List of substations forced to split.
    cong_limit : float
        Loading-ratio threshold used to identify congested lines.
    print_result : bool
        If True, print congestion and switching results.

    Returns
    -------
    dict or str
        Dictionary of OSS results if solved optimally; otherwise, returns "Infeasible".
    """
    
    #======  data
    
    Sbase=data['Sbase']
    Max_MIPGap=data['Max_MIPGap']
    Max_timelimit=data['Max_timelimit'] #=300 #900
    Bus=data['Bus']    # for b in Bus
    branch=data['branch']
    Lines=data['Lines']
    line=data['line']
    Lines1D=data['Lines1D']
    L2B=data['L2B']
    DemandSet=data['Demandset']     
    D2B=data['D2B']                 
    G=data['G']            
    G2B=data['G2B']    

    if Substation_nodes==[]:
        Bus_sub = Bus.copy()
    else:
        Bus_sub = Substation_nodes.copy()

    
    
    
    #======
    if res_DCOPF is None:
        print('we need DC solution.')



    model = modelX.clone()
    
    if Pd_instance is not None:
        for d_ind, d in enumerate(data['Demandset']):
            model.Pd_param[d] = Pd_instance[d_ind]
    
    if res_DCOPF is not None:
        for g in G:
            model.Pg_DCOPF[g] = res_DCOPF['Pg'][g]  

    if limit_new is not None:
        for l_ind,(l,i,j) in enumerate(Lines1D):
            model.limit[l,i,j] = limit_new[l_ind]
            model.limit[l,j,i] = limit_new[l_ind] #bi-directional lines
        
        #new cong sg and cost
        Cong_sg_data , _ = compute_congestion_segments(line, limit_new, data['Seg_num'],data['ds_loading'],data['gamma'])
        for l,sg in model.LineSeg:
            model.cong_df[l,sg] = Cong_sg_data.loc[(l, sg), 'df']
            model.cong_cost_s[l,sg] = Cong_sg_data.loc[(l, sg), 's']


    if gen_cost_new is not None:
        for g_ind,g in enumerate(G):
            model.gen_cost_b[g] = gen_cost_new[g_ind]


    # === Update outage set and reconstruct flow constraint
    model.outage_lines.clear()
    model.outage_lines.update(lines_outage)

    model.flow_eq.clear()
    model.flow_eq._constructed = False
    model.flow_eq.construct()

    model.zero_flow_eq.clear()
    model.zero_flow_eq._constructed = False
    model.zero_flow_eq.construct()



    if candidate_bus==None:
        candidate_bus = Substation_nodes.copy() #all are candidates for switching, however, if specified, those are only considered


    if candidate_bus is not None:
        model.fix_zb = pyo.ConstraintList()
        model.fix_zg = pyo.ConstraintList()
        model.fix_zd = pyo.ConstraintList()
        model.fix_zli = pyo.ConstraintList()
        for b in Bus_sub:
            if b not in candidate_bus:
                model.fix_zb.add(model.z_bus[b] == 1)
                for g, i in G2B.select('*',b):
                    model.fix_zg.add(model.z_g[g] == 0)
                for d, i in D2B.select('*',b):
                    model.fix_zd.add(model.z_d[d] == 0)
                for l, i, j in Lines.select('*',b,'*'):
                    model.fix_zli.add(model.z_li[l, i, j] == 0)

    #force split
    model.force_split_constr = pyo.ConstraintList()
    for b in force_split:
        if b in Bus_sub:
            model.force_split_constr.add(model.z_bus[b] == 0)


    
    # === Solve
    try:
        start_time = time.time()
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        result = SolverFactory("gurobi").solve(model, tee=False, options={"OutputFlag": 0})
        end_time = time.time()
        ex_time = end_time - start_time

        # === Check feasibility
        if (result.solver.termination_condition != pyo.TerminationCondition.optimal) or (result.solver.status != SolverStatus.ok):
            return "Infeasible"
        
        z_bus_open=[]
        for b in Bus_sub:
            if model.z_bus[b].value==0:
                z_bus_open+=[b]
        
        # === Flow values
        if limit_new is not None:
            limit = limit_new.copy()
        else:
            limit = data['branch']['limit'].to_numpy()[:len(data['line'])] 
        
        flows = np.zeros(len(line))
        Pflow_df = pd.DataFrame(columns=['flow'], index=data['branch1D'].index) 

        for l_ind, (l, i, j) in enumerate(Lines1D):
            flows[l_ind] = model.Pflow[l, i, j].value
            Pflow_df.loc[(l,i,j),'flow']=model.Pflow[l,i,j].value


        loading_ratio = flows / limit
        overloaded_mask = np.abs(loading_ratio) >= cong_limit
        Num_cong = np.sum(overloaded_mask)


        # === Flows on lines in service
        Lines_in_service = [(l, i, j) for l, i, j in Lines1D if l not in lines_outage]
        Pflow_top = np.zeros(len(Lines_in_service))
        for idx, (l, i, j) in enumerate(Lines_in_service):
            Pflow_top[idx] = model.Pflow[l, i, j].value

        # === Print if needed
        if print_result:
            # print(f"\t GenCost: {model.obj:.3f}")
            print(f"\t Num of congested lines: {Num_cong}")
            print('\t TotalCongCost: %.3f'%model.TotalCongCost.value)
            print('\t Z_bus open is %s'%z_bus_open)

        return {
            # 'GenCost': pyo.value(model.GenCost),
            'time': ex_time,
            'Pflow': flows,
            'Pflow_top': Pflow_top,
            'Pflow_df':Pflow_df,
            'Num_cong': Num_cong,
            'z_bus_open':z_bus_open,
            'overloaded_mask':overloaded_mask
        }
    except:
        return "Infeasible"


    
    
    


# # Example
# data = read_data_AC('IEEE_14_bus_Data_PGLib_ACOPF.xlsx',DemFactor=2.0,LineLimit=1.0,Seg_num=5,gamma=10)
# Pd = data['Pdemand']['Pd'].to_numpy()
# limits = data['branch1D']['limit'].to_numpy()
# gcost = 2*data['Gen_data']['b'].to_numpy()
# modelX = Create_DC_OPF_Pyomo(data)
# resDC = Solve_DC_OPF_Pyomo(data=data,modelX=modelX,Pd_instance=Pd,lines_outage=[], limit_new= limits, gen_cost_new=gcost ,print_result=True)
# modelOSS= Create_Cong_DC_OSS_Pyomo(data,Max_Sw_bus=0)
# res= Solve_Cong_DC_OSS_Pyomo(data=data,modelX=modelOSS,Pd_instance=Pd,lines_outage=[],candidate_bus=['b4'], limit_new= limits, gen_cost_new=gcost,res_DCOPF=resDC,print_result=True)





# %%
    
def Create_ML_Cong_DC_OSS_Pyomo(
    data,
    Substation_nodes=None,
    Max_Sw_bus=0,
    LineLimit=1.0,   # you can relax the line limit to allow force-switching
    Zinitial=None,
    Zfixdict=None,
):
    """
    Create a machine-learning-assisted congestion-management DC optimal substation
    switching (OSS) Pyomo model.

    This formulation is designed for topology optimization using known net power injections instead of explicit generator and demand variables.
    The model minimizes congestion and infeasibility penalties while considering
    busbar splitting, line-flow constraints, and DC power-flow equations.

    Parameters
    ----------
    data : dict
        Input network and system data dictionary.
    Substation_nodes : list, optional
        List of buses modeled as substations with switchable busbars.
        If None, all buses are modeled as substations.
    Max_Sw_bus : int
        Maximum number of substations allowed to split.
    LineLimit : float
        Scaling factor applied to line-flow limits.
    Zinitial : dict, optional
        Initial values for binary switching variables.
    Zfixdict : dict, optional
        Dictionary for fixing switching configurations.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        ML-assisted congestion-management OSS Pyomo model.
    """
    
    #======  data
    
    
    Bus=data['Bus']    # for b in Bus
    busbar=data['busbar']
    branch=data['branch']
    Lines=data['Lines']
    line=data['line']
    Lines1D=data['Lines1D']
    NumberL2B=data['NumberL2B']            
    Seg=data['Seg']
    Cong_sg_data=data['Cong_sg_data']
    BigM_b=data['BigM_b']
    Maxdelta=data['Maxdelta'] 
    if Substation_nodes is None:
        print('All nodes are modeled as substations.')
        Bus_sub = Bus.copy()
    else:
        Bus_sub = Substation_nodes.copy()
    
    
    model = pyo.ConcreteModel()

    # === Sets
    model.Bus = pyo.Set(initialize=Bus)
    model.busbar = pyo.Set(initialize=busbar)
    model.Lines = pyo.Set(initialize=Lines, dimen=3)
    model.Lines1D = pyo.Set(initialize=Lines1D, dimen=3)
    model.line = pyo.Set(initialize=line)
    model.Seg = pyo.Set(initialize=Seg)

    # === Substation definitions
    Bus_non_sub = [b for b in Bus if b not in Bus_sub]

    Lines_sub = [(l, i, j) for sub in Bus_sub for l, i, j in Lines.select('*', sub, '*')]

    model.Bus_sub = pyo.Set(initialize=Bus_sub)
    model.Bus_non_sub = pyo.Set(initialize=Bus_non_sub)
    
    model.Lines_sub = pyo.Set(initialize=Lines_sub, dimen=3)
    model.LineSeg = pyo.Set(initialize=[(l, sg) for l in line for sg in Seg], dimen=2)

    # Outage line set (mutable so you can update it later)
    model.outage_lines = pyo.Set(initialize=[],dimen=1)

    # === Params
    Pinj_init = {b: 0 for b in Bus}
    model.P_inj = pyo.Param(model.Bus, initialize=Pinj_init, mutable=True)

    limit_init = {(l,i,j): branch.loc[(l, i, j), 'limit'] for l, i, j in Lines}
    model.limit = pyo.Param(model.Lines, initialize=limit_init, mutable=True)

    cong_cost_init = {(l,sg): Cong_sg_data.loc[(l, sg), 's']  for l,sg in model.LineSeg  }
    model.cong_cost_s = pyo.Param(model.LineSeg, initialize=cong_cost_init, mutable=True)

    cong_df_init = {(l,sg): Cong_sg_data.loc[(l, sg), 'df']  for l,sg in model.LineSeg}
    model.cong_df = pyo.Param(model.LineSeg, initialize=cong_df_init, mutable=True)

    


    # === Variables
    model.Pflow = pyo.Var(model.Lines)
    model.Pflow_li = pyo.Var(model.Lines_sub, model.busbar)

    model.delta_bi = pyo.Var(model.Bus_sub, model.busbar)
    model.delta_li = pyo.Var(model.Lines)

    model.z_bus = pyo.Var(model.Bus_sub, domain=pyo.Binary)
    model.z_li = pyo.Var(model.Lines_sub, domain=pyo.Binary)
    
    model.sp = pyo.Var(model.Bus_non_sub, domain=pyo.NonNegativeReals)
    model.sn = pyo.Var(model.Bus_non_sub, domain=pyo.NonNegativeReals)
    model.sp_bi = pyo.Var(model.Bus_sub, model.busbar, domain=pyo.NonNegativeReals)
    model.sn_bi = pyo.Var(model.Bus_sub, model.busbar, domain=pyo.NonNegativeReals)
    model.InfCost = pyo.Var(domain=pyo.NonNegativeReals)

    model.Pflow_ps = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.Pflow_ng = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.CongCost = pyo.Var(model.line, domain=pyo.NonNegativeReals)
    model.TotalCongCost = pyo.Var(domain=pyo.NonNegativeReals)
    # model.GenCost = pyo.Var(domain=pyo.NonNegativeReals)
    model.OF = pyo.Var(domain=pyo.NonNegativeReals)


    # Initial values
    if Zinitial is None:
        for b in Bus_sub:
            model.z_bus[b].value = 1
        for l, i, j in Lines_sub:
            model.z_li[l, i, j].value = 0
       
    else:
        for b in Bus_sub:
            model.z_bus[b].value = Zinitial['bus'][b]
        for l, i, j in Lines_sub:
            model.z_li[l, i, j].value = Zinitial['l_i'][l, i, j]


    # Constraints for 2-line rule
    def eq_2line_busbar2_rule(m, b):
        return 2 * (1 - m.z_bus[b]) <= sum(m.z_li[l, i, j] for (l, i, j) in Lines.select('*',b,'*'))
    model.eq_2line_busbar2 = Constraint(model.Bus_sub, rule=eq_2line_busbar2_rule)

    def eq_2line_busbar1_rule(m, b):
        return 2 * (1 - m.z_bus[b]) <= sum(1 - m.z_li[l, i, j] for (l, i, j) in Lines.select('*',b,'*'))
    model.eq_2line_busbar1 = Constraint(model.Bus_sub, rule=eq_2line_busbar1_rule)

    def eq_2line_busbar3_rule(m, b):
        if NumberL2B[b] <= 3:
            return m.z_bus[b] == 1
        return Constraint.Skip
    model.eq_2line_busbar3 = Constraint(model.Bus_sub, rule=eq_2line_busbar3_rule)

    def max_sw_bus_rule(m):
        if len(m.Bus_sub) == 0:
            return Constraint.Skip  # no constraint needed
        return sum(1 - m.z_bus[b] for b in m.Bus_sub) <= Max_Sw_bus

    model.eq_MaxSw_bus = pyo.Constraint(rule=max_sw_bus_rule)

    model.eq_blFe = ConstraintList()
    for b in Bus_sub:
        for l, i, j in Lines.select('*',b,'*'):
            model.eq_blFe.add(model.z_bus[b] - 1 + model.z_li[l, i, j] <= 0)

    
    # DC Equations
    def flow_eq_with_outage(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        return m.Pflow[l, i, j] == (m.delta_li[l,i,j] - m.delta_li[l,j,i]) / data['branch'].loc[(l, i, j), 'x']
    model.flow_eq = pyo.Constraint(model.Lines, rule=flow_eq_with_outage)

    def zero_flow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Pflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_flow_eq = pyo.Constraint(model.Lines, rule=zero_flow_eq)

    
    model.pij_max = pyo.Constraint(model.Lines, rule=lambda m, l, i, j:
        m.Pflow[l, i, j] <= m.limit[l,i,j]*LineLimit )

    model.pij_min = pyo.Constraint(model.Lines, rule=lambda m, l, i, j:
        -m.limit[l,i,j]*LineLimit <= m.Pflow[l, i, j] )


    def eq_flow1min_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit <= m.Pflow_li[l, i, j, 'busbar1']
    model.eq_flow1min = pyo.Constraint(model.Lines_sub, rule=eq_flow1min_rule)

    def eq_flow1max_rule(m, l, i, j):
        return m.Pflow_li[l, i, j, 'busbar1'] <= (1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit
    model.eq_flow1max = pyo.Constraint(model.Lines_sub, rule=eq_flow1max_rule)

    def eq_flow2min_rule(m, l, i, j):
        return -m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit <= m.Pflow_li[l, i, j, 'busbar2']
    model.eq_flow2min = pyo.Constraint(model.Lines_sub, rule=eq_flow2min_rule)

    def eq_flow2max_rule(m, l, i, j):
        return m.Pflow_li[l, i, j, 'busbar2'] <= m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit
    model.eq_flow2max = pyo.Constraint(model.Lines_sub, rule=eq_flow2max_rule)

    def eq_flow_sum_rule(m, l, i, j):
        return m.Pflow[l, i, j] == m.Pflow_li[l, i, j, 'busbar1'] + m.Pflow_li[l, i, j, 'busbar2']
    model.eq_Pflow = pyo.Constraint(model.Lines_sub, rule=eq_flow_sum_rule)


    #delta
    def eq_delta_bus1_rule(m, b):
        return -BigM_b * (1 - m.z_bus[b]) <= m.delta_bi[b, 'busbar1'] - m.delta_bi[b, 'busbar2']
    model.eq_delta_bus1 = pyo.Constraint(model.Bus_sub, rule=eq_delta_bus1_rule)

    def eq_delta_bus2_rule(m, b):
        return m.delta_bi[b, 'busbar1'] - m.delta_bi[b, 'busbar2'] <= BigM_b * (1 - m.z_bus[b])
    model.eq_delta_bus2 = pyo.Constraint(model.Bus_sub, rule=eq_delta_bus2_rule)

    # Delta difference bounds for lines
    def eq_delta_lb1Frmin_rule(m, l, i, j):
        return -m.z_li[l, i, j] * Maxdelta <= m.delta_li[l, i, j] - m.delta_bi[i, 'busbar1']
    model.eq_delta_lb1Frmin = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb1Frmin_rule)

    def eq_delta_lb1Frmax_rule(m, l, i, j):
        return m.delta_li[l, i, j] - m.delta_bi[i, 'busbar1'] <= m.z_li[l, i, j] * Maxdelta
    model.eq_delta_lb1Frmax = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb1Frmax_rule)

    def eq_delta_lb2Frmin_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * Maxdelta <= m.delta_li[l, i, j] - m.delta_bi[i, 'busbar2']
    model.eq_delta_lb2Frmin = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb2Frmin_rule)

    def eq_delta_lb2Frmax_rule(m, l, i, j):
        return m.delta_li[l, i, j] - m.delta_bi[i, 'busbar2'] <= (1 - m.z_li[l, i, j]) * Maxdelta
    model.eq_delta_lb2Frmax = pyo.Constraint(model.Lines_sub, rule=eq_delta_lb2Frmax_rule)

    #delta i
    model.eqDelta = pyo.ConstraintList()
    for b in Bus_non_sub:
        lmin, b, j1 = Lines.select('*', b, '*')[0] #first line
        for l, i, j in Lines.select('*', b, '*'):
            model.eqDelta.add(
                    model.delta_li[lmin, b, j1] == model.delta_li[l, i, j])
    
    # Reference angle constraint
    l0, i0, j0 = Lines[0]
    model.ref_bus_angle = pyo.Constraint(expr=model.delta_li[l0, i0, j0] == 0)



    # === Balance constraint for substation buses
    def eq_balance_sub_rule(m, b, mbar):
        inj_sum = m.P_inj[b] if mbar == 'busbar1' else 0
        flow_sum = sum(m.Pflow_li[l, i, j, mbar] for l, i, j in Lines.select('*',b,'*'))
        return inj_sum + m.sp_bi[b,mbar] - m.sn_bi[b,mbar] == flow_sum
    model.eq_balance_sub = pyo.Constraint(model.Bus_sub, model.busbar, rule=eq_balance_sub_rule)

    # === Balance constraint for non-substation buses
    def eq_balance_nonsub_rule(m, b):
        inj_sum = m.P_inj[b] 
        flow_sum = sum(m.Pflow[l, i, j] for l, i, j in Lines.select('*',b,'*'))
        return inj_sum + m.sp[b] - m.sn[b] ==  flow_sum
    model.eq_balance_nonsub = pyo.Constraint(model.Bus_non_sub, rule=eq_balance_nonsub_rule)

   
    # === Congestion power flow decomposition
    def eqPflowSumCong_rule(m, l, i, j):
        return m.Pflow[l, i, j] == sum(m.Pflow_ps[l, sg] for sg in m.Seg) - sum(m.Pflow_ng[l, sg] for sg in m.Seg)
    model.eqPflowSumCong = pyo.Constraint(model.Lines1D, rule=eqPflowSumCong_rule)

    # === Upper bounds on Pflow_ps
    def eqPflow_ps_SgMax_rule(m, l, sg):
        return m.Pflow_ps[l, sg] <= model.cong_df[l,sg]
    model.eqPflow_ps_SgMax = pyo.Constraint(model.line, model.Seg, rule=eqPflow_ps_SgMax_rule)

    # === Upper bounds on Pflow_ng
    def eqPflow_ng_SgMax_rule(m, l, sg):
        return m.Pflow_ng[l, sg] <= model.cong_df[l,sg]  
    model.eqPflow_ng_SgMax = pyo.Constraint(model.line, model.Seg, rule=eqPflow_ng_SgMax_rule)

    # === Congestion cost per line
    def eqCongCost_rule(m, l, i, j):
        return m.CongCost[l] == sum(m.Pflow_ps[l, sg] * m.cong_cost_s[l,sg] for sg in m.Seg) + \
                                sum(m.Pflow_ng[l, sg] * m.cong_cost_s[l,sg] for sg in m.Seg)
    model.eqCongCost = pyo.Constraint(model.Lines1D, rule=eqCongCost_rule)

    # === Total congestion cost
    model.eqTotalCongCost = pyo.Constraint(expr=model.TotalCongCost == sum(model.CongCost[l] for l in model.line))

    # === Link total cost to objective
    model.Eq_OF = pyo.Constraint(expr=model.OF == model.TotalCongCost + model.InfCost)   
    
    model.Eq_Inf = pyo.Constraint(expr=model.InfCost == 1e+4*sum(model.sp[b] + model.sn[b] for b in model.Bus_non_sub) 
                                + 1e+4*sum(model.sp_bi[b,i] + model.sn_bi[b,i] for b in model.Bus_sub for i in model.busbar ) )   

    # === Objective function
    model.obj = pyo.Objective(expr=model.OF, sense=pyo.minimize)

    return model






# %%

def Solve_ML_Cong_DC_OSS_Pyomo(
    data,
    modelX,
    Substation_nodes=None,
    lines_outage=[],
    X_inj=None,
    limit_new=None,
    candidate_bus=None,
    force_split=[],
    cong_limit=0.95,
    print_result=False,
):
    """
    Solve the machine-learning-assisted congestion-management DC optimal
    substation switching (OSS) model.

    This function solves the ML-based OSS formulation using externally provided
    nodal injections. The model updates injection values, line limits, congestion
    segments, and switching constraints, then minimizes congestion and
    infeasibility penalties using topology optimization.

    Parameters
    ----------
    data : dict
        Input network and system data dictionary.
    modelX : pyomo.environ.ConcreteModel
        Base ML-assisted DC-OSS Pyomo model to be cloned and solved.
    Substation_nodes : list, optional
        List of buses modeled as switchable substations.
        If None, all buses are modeled as substations.
    lines_outage : list
        List of line IDs that are out of service.
    X_inj : array-like, optional
        External nodal injection vector used in the ML-assisted formulation.
    limit_new : array-like, optional
        Updated line limits for one-directional lines.
    candidate_bus : list, optional
        Candidate substations allowed to split.
    force_split : list
        List of substations forced to split.
    cong_limit : float
        Loading-ratio threshold used to identify congested lines.
    print_result : bool
        If True, print congestion and switching results.

    Returns
    -------
    dict or str
        Dictionary of solved OSS results if successful; otherwise returns
        "Infeasible".
    """
    
    #======  data
    
    
    Bus=data['Bus']    # for b in Bus
    Lines=data['Lines']
    line=data['line']
    Lines1D=data['Lines1D']
    
    if Substation_nodes is None:
        Bus_sub = Bus.copy()
    else:
        Bus_sub = Substation_nodes.copy()

    model = modelX.clone()
    
    if X_inj is not None:
        for b_ind, b in enumerate(data['Bus']):
            model.P_inj[b] = X_inj[b_ind]
    

    if limit_new is not None:
        for l_ind,(l,i,j) in enumerate(Lines1D):
            model.limit[l,i,j] = limit_new[l_ind]
            model.limit[l,j,i] = limit_new[l_ind] #bi-directional lines
        
        #new cong sg and cost
        Cong_sg_data , _ = compute_congestion_segments(line, limit_new, data['Seg_num'],data['ds_loading'],data['gamma'])
        for l,sg in model.LineSeg:
            model.cong_df[l,sg] = Cong_sg_data.loc[(l, sg), 'df']
            model.cong_cost_s[l,sg] = Cong_sg_data.loc[(l, sg), 's']



    # === Update outage set and reconstruct flow constraint
    # model.outage_lines.clear()
    # model.outage_lines.update(lines_outage)

    # model.flow_eq.clear()
    # model.flow_eq._constructed = False
    # model.flow_eq.construct()

    # model.zero_flow_eq.clear()
    # model.zero_flow_eq._constructed = False
    # model.zero_flow_eq.construct()

    

    if candidate_bus is None:
        candidate_bus = Substation_nodes.copy() #all are candidates for switching, however, if specified, those are only considered


    if candidate_bus is not None:
        model.fix_zb = pyo.ConstraintList()
        # model.fix_zg = pyo.ConstraintList()
        # model.fix_zd = pyo.ConstraintList()
        model.fix_zli = pyo.ConstraintList()
        for b in model.Bus_sub:
            if b not in candidate_bus:
                model.fix_zb.add(model.z_bus[b] == 1)
                for l, i, j in Lines.select('*',b,'*'):
                    model.fix_zli.add(model.z_li[l, i, j] == 0)

    #force split
    model.force_split_constr = pyo.ConstraintList()
    for b in force_split:
        if b in Bus_sub:
            model.force_split_constr.add(model.z_bus[b] == 0)


    
    # === Solve
    try:
        start_time = time.time()
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        result = SolverFactory("gurobi").solve(model, tee=False, options={"OutputFlag": 0,'TimeLimit': 36000})
        end_time = time.time()
        ex_time = end_time - start_time

        # === Check feasibility
        # if result.solver.termination_condition in [TerminationCondition.infeasible, TerminationCondition.unbounded]:
        #     return "Infeasible"
        

        z_bus_open=[]
        for b in Bus_sub:
            if model.z_bus[b].value==0:
                z_bus_open+=[b]
        
        # === Flow values
        if limit_new is not None:
            limit = limit_new.copy()
        else:
            limit = data['branch']['limit'].to_numpy()[:len(data['line'])] 
        flows = np.zeros(len(line))
        for l_ind, (l, i, j) in enumerate(Lines1D):
            flows[l_ind] = model.Pflow[l, i, j].value


        loading_ratio = flows / limit
        overloaded_mask = np.abs(loading_ratio) >= cong_limit
        Num_cong = np.sum(overloaded_mask)
        
        # === Flows on lines in service
        Lines_in_service = [(l, i, j) for l, i, j in Lines1D if l not in lines_outage]
        Pflow_top = np.zeros(len(Lines_in_service))
        for idx, (l, i, j) in enumerate(Lines_in_service):
            Pflow_top[idx] = model.Pflow[l, i, j].value

        # === Print if needed
        if print_result:
            # print(f"\t GenCost: {model.obj:.3f}")
            print(f"\t Num of congested lines: {Num_cong}")
            print('\t TotalCongCost: %.3f'%model.TotalCongCost.value)
            print('\t Z_bus open is %s'%z_bus_open)
            print('\t Inf is %.2f' % model.InfCost.value)

        return {
            # 'GenCost': pyo.value(model.GenCost),
            'time': ex_time,
            'Pflow': flows,
            'Pflow_top': Pflow_top,
            'Num_cong': Num_cong,
            'z_bus_open':z_bus_open,
            'CongCost':model.TotalCongCost.value
        }
    except:
        return "Infeasible"


    