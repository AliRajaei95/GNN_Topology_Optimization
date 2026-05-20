"""
AC-based optimal substation switching and congestion-management utilities.

This module contains data-loading helpers, graph/network utilities,
AC-OPF formulations, feasibility-restoration models, and Pyomo-based
AC optimal substation switching (OSS) formulations for congestion management
and topology optimization in power systems.

The implementations include:
    - grid input data are in the DC-OSS file
    - AC optimal power flow (AC-OPF)
    - Feasibility-restoration AC-OPF
    - Machine-learning-assisted AC-OSS formulations

Developed for power-system operation and topology-control studies using
Pyomo and Gurobi optimization frameworks.

Tested with Python 3.12 and Gurobi 12.0.
"""

# %%
# Libraries

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

# Given AC OPF
def Create_AC_feasible_OPF(data):
    """
    Substaion nodes are nodes that have more than 4 lines in the original dataset, so they're modelled as substation nodes, others are single electrical nodes.
    """
    
    #======  data
    
    
    Bus=data['Bus']    # for b in Bus
    branch=data['branch']
    Lines=data['Lines']
    #======
   
    
    
    model = pyo.ConcreteModel()

    # === Sets
    model.Bus = pyo.Set(initialize=Bus)
    
    model.Lines = pyo.Set(initialize=Lines, dimen=3)
    
    # Outage line set (mutable so you can update it later)
    model.outage_lines = pyo.Set(initialize=[],dimen=1)

    # === Params
    Pinj_init = {b: 0 for b in Bus}
    model.P_inj = pyo.Param(model.Bus, initialize=Pinj_init, mutable=True)

    Qinj_init = {b: 0 for b in Bus}
    model.Q_inj = pyo.Param(model.Bus, initialize=Qinj_init, mutable=True)


    limit_init = {(l,i,j): branch.loc[(l, i, j), 'limit'] for l, i, j in Lines}
    model.limit = pyo.Param(model.Lines, initialize=limit_init, mutable=True)

    vmin=0.9; vmax=1.1;

    # === Variables
    model.Pflow = pyo.Var(model.Lines)
    model.Qflow = pyo.Var(model.Lines)
    model.V2 = pyo.Var(model.Bus,domain=pyo.NonNegativeReals,initialize=1)
    model.theta = pyo.Var(model.Bus,initialize=0)

    model.sp_pos = pyo.Var(model.Bus, domain=pyo.NonNegativeReals)
    model.sp_neg = pyo.Var(model.Bus, domain=pyo.NonNegativeReals)
    model.sq_pos = pyo.Var(model.Bus, domain=pyo.NonNegativeReals)
    model.sq_neg = pyo.Var(model.Bus, domain=pyo.NonNegativeReals)

    model.Pinj_new = pyo.Var(model.Bus)
    model.Qinj_new = pyo.Var(model.Bus)
    
    model.InfCost = pyo.Var(domain=pyo.NonNegativeReals)

    def eqPinj(model,b):
        return model.Pinj_new[b] == model.P_inj[b] + model.sp_pos[b] - model.sp_neg[b]
    model.eqPinj=Constraint(Bus,rule=eqPinj)

    def eqQinj(model,b):
        return model.Qinj_new[b] == model.Q_inj[b] + model.sq_pos[b] - model.sq_neg[b]
    model.eqQinj=Constraint(Bus,rule=eqQinj)


    def eqPbalance(model,b):
        return model.Pinj_new[b] == sum(model.Pflow[l,i,j] for l,i,j in Lines.select('*',b,'*'))
    model.eqPbalance=Constraint(Bus,rule=eqPbalance)

    def eqQbalance(model,b):
        return model.Qinj_new[b] == sum(model.Qflow[l,i,j] for l,i,j in Lines.select('*',b,'*'))
    model.eqQbalance=Constraint(Bus,rule=eqQbalance)


    # LP AC equations

    def eqPij_rule(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        g_ij = branch.loc[(l,i,j)]['g_ij']
        b_ij = branch.loc[(l,i,j)]['b_ij']
        return m.Pflow[l,i,j] == 0.5*g_ij*(m.V2[i] - m.V2[j]) - b_ij*(m.theta[i] - m.theta[j])
    model.eqPij = pyo.Constraint(model.Lines, rule=eqPij_rule)

    def eqQij_rule(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        g_ij = branch.loc[(l,i,j)]['g_ij']
        b_ij = branch.loc[(l,i,j)]['b_ij']
        return m.Qflow[l,i,j] == -0.5*b_ij*(m.V2[i] - m.V2[j]) - g_ij*(m.theta[i] - m.theta[j])
    model.eqQij = pyo.Constraint(model.Lines, rule=eqQij_rule)


    def zero_Pflow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Pflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_Pflow_eq = pyo.Constraint(model.Lines, rule=zero_Pflow_eq)

    def zero_Qflow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Qflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_Qflow_eq = pyo.Constraint(model.Lines, rule=zero_Qflow_eq)


    def eqVmax(model, i):
        return model.V2[i] <= vmax**2
    model.eqVmax = pyo.Constraint(Bus, rule=eqVmax)

    def eqVmin(model, i):
        return vmin**2 <= model.V2[i]
    model.eqVmin = pyo.Constraint(Bus, rule=eqVmin)


    # Branch flow limits (P-Q combined)
    beta = 0
    def eqPij1(model, l, i, j):
        return model.Pflow[l,i,j] + np.tan(np.pi/6)*model.Qflow[l,i,j] <= (1-beta)*branch.loc[(l,i,j)]['limit']
    model.eqPij1 = pyo.Constraint(Lines, rule=eqPij1)

    def eqPij2(model, l, i, j):
        return model.Pflow[l,i,j] - np.tan(np.pi/6)*model.Qflow[l,i,j] <= (1-beta)*branch.loc[(l,i,j)]['limit']
    model.eqPij2 = pyo.Constraint(Lines, rule=eqPij2)

    def eqPij3(model, l, i, j):
        return model.Pflow[l,i,j] + np.tan(np.pi/6)*model.Qflow[l,i,j] >= (beta-1)*branch.loc[(l,i,j)]['limit']
    model.eqPij3 = pyo.Constraint(Lines, rule=eqPij3)

    def eqPij4(model, l, i, j):
        return model.Pflow[l,i,j] - np.tan(np.pi/6)*model.Qflow[l,i,j] >= (beta-1)*branch.loc[(l,i,j)]['limit']
    model.eqPij4 = pyo.Constraint(Lines, rule=eqPij4)

    
    def eqtheta(model):
        return model.theta[Bus[0]]==0
    model.eqtheta=Constraint(rule=eqtheta)


    # === Link total cost to objective
    model.Eq_Inf = pyo.Constraint(expr=model.InfCost == sum(model.sp_pos[b]+ model.sp_neg[b]+model.sq_pos[b] + model.sq_neg[b]    for b in model.Bus)  )
   
    # === Objective function
    model.obj = pyo.Objective(expr=model.InfCost, sense=pyo.minimize)

    return model

# %%

def Solve_AC_feasible_OPF(data,
                          modelX,
                          lines_outage=[],
                          X_Pinj=None,
                          X_Qinj=None,
                          limit_new=None,
                          cong_limit=0.95,
                          print_result=False):
    """
    Solve the AC feasibility-restoration OPF model.

        The function clones a base model, updates active/reactive injections and
        optional line limits, solves the feasibility problem, and returns the adjusted
        nodal injections.

        Parameters
        ----------
        data : dict
            Input network and system data dictionary.
        modelX : pyomo.environ.ConcreteModel
            Base AC feasibility model to clone and solve.
        lines_outage : list
            Line IDs removed from service.
        X_Pinj : array-like, optional
            Active nodal injection vector.
        X_Qinj : array-like, optional
            Reactive nodal injection vector.
        limit_new : array-like, optional
            Updated one-directional line limits.
        cong_limit : float
            Loading threshold used when checking congestion.
        print_result : bool
            If True, print infeasibility cost.

        Returns
        -------
        dict or str
            Feasibility results, or "Infeasible" if the model fails.
    """

    
    #======  data
    Bus=data['Bus']    # for b in Bus
    Lines1D=data['Lines1D']
    
    model = modelX.clone()
    
    if X_Pinj is not None:
        for b_ind, b in enumerate(data['Bus']):
            model.P_inj[b] = X_Pinj[b_ind]
    
    if X_Qinj is not None:
        for b_ind, b in enumerate(data['Bus']):
            model.Q_inj[b] = X_Qinj[b_ind]
      

    if limit_new is not None:
        for l_ind,(l,i,j) in enumerate(Lines1D):
            model.limit[l,i,j] = limit_new[l_ind]
            model.limit[l,j,i] = limit_new[l_ind] #bi-directional lines
        

    # === Update outage set and reconstruct flow constraint
    # model.outage_lines.clear()
    # model.outage_lines.update(lines_outage)

    # model.flow_eq.clear()
    # model.flow_eq._constructed = False
    # model.flow_eq.construct()

    # model.zero_flow_eq.clear()
    # model.zero_flow_eq._constructed = False
    # model.zero_flow_eq.construct()

    
    # === Solve
    try:
        start_time = time.time()
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        result = SolverFactory("gurobi").solve(model, tee=False, options={"OutputFlag": 0})
        end_time = time.time()
        ex_time = end_time - start_time

        # === Check feasibility
        if result.solver.termination_condition in [TerminationCondition.infeasible, TerminationCondition.unbounded]:
            return "Infeasible"
        
        pinj_new = np.zeros(len(Bus))
        qinj_new = np.zeros(len(Bus))

        for b_ind, b in enumerate(data['Bus']):
            pinj_new[b_ind] = model.Pinj_new[b].value
            qinj_new[b_ind] = model.Qinj_new[b].value
            

        # === Print if needed
        if print_result:
            # print(f"\t GenCost: {model.obj:.3f}")
            print('\t InfCost: %.3f'%model.InfCost.value)

        return {
            'time': ex_time,
            'pinj':pinj_new,
            'qinj':qinj_new
        }
    except:
        return "Infeasible"







# %% AC OPF

def Create_AC_OPF_Pyomo(data):

    """
    Create a linearized AC optimal power flow Pyomo model.

        The model minimizes generation cost subject to active/reactive power balance,
        linearized AC branch-flow equations, voltage limits, generator limits, and
        polygonal apparent-power flow limits.

        Parameters
        ----------
        data : dict
            Input network and system data dictionary.

        Returns
        -------
        model : pyomo.environ.ConcreteModel
            Pyomo AC-OPF model.
    """
    

    #======  data
    Bus=data['Bus'] 
    branch=data['branch']
    Lines=data['Lines']                
    Gen_data=data['Gen_data']
    G=data['G']            
    G2B=data['G2B']             
    
    model = pyo.ConcreteModel()

    # === Sets
    model.G = pyo.Set(initialize=G)
    model.Bus = pyo.Set(initialize=Bus)
    model.Lines = pyo.Set(initialize=Lines, dimen=3)

    # Outage line set (mutable so you can update it later)
    model.outage_lines = pyo.Set(initialize=[],dimen=1)

    # === Params
    Pd_init = {d: data['Pdemand'].loc[d, 'Pd'] for d in data['Demandset']}
    model.DemandSet = pyo.Set(initialize=data['Demandset'])
    model.Pd = pyo.Param(model.DemandSet, initialize=Pd_init, mutable=True)

    Qd_init = {d: data['Pdemand'].loc[d, 'Qd'] for d in data['Demandset']}
    model.DemandSet = pyo.Set(initialize=data['Demandset'])
    model.Qd = pyo.Param(model.DemandSet, initialize=Qd_init, mutable=True)

    limit_init = {(l,i,j): branch.loc[(l, i, j), 'limit'] for l, i, j in Lines}
    model.limit = pyo.Param(model.Lines, initialize=limit_init, mutable=True)

    gen_cost_init = {g: Gen_data.loc[g, 'b'] for g in G}
    model.gen_cost_b = pyo.Param(model.G, initialize=gen_cost_init, mutable=True)

    
    # === Variables
    model.Pg = pyo.Var(model.G, domain=pyo.NonNegativeReals)
    model.Qg = pyo.Var(model.G, domain=pyo.NonNegativeReals)
    model.Pinj = pyo.Var(model.Bus)
    model.Qinj = pyo.Var(model.Bus)
    

    model.Pflow = pyo.Var(model.Lines)
    model.Qflow = pyo.Var(model.Lines)
    

    model.V2 = pyo.Var(model.Bus,domain=pyo.NonNegativeReals,initialize=1)
    model.theta = pyo.Var(model.Bus,initialize=0)


    model.delta = pyo.Var(model.Bus)
    model.Pflow = pyo.Var(model.Lines)

    vmin=0.9; vmax=1.1; 

    # === Constraints

    def eqPinj(model,b):
        return model.Pinj[b] == sum( model.Pg[g] for g,b in G2B.select('*',b) ) - sum(model.Pd[d] for d, bb in data['D2B'] if bb == b)
    model.eqPinj=Constraint(Bus,rule=eqPinj)

    def eqQinj(model,b):
        return model.Qinj[b] == sum( model.Qg[g] for g,b in G2B.select('*',b) ) - sum(model.Qd[d] for d, bb in data['D2B'] if bb == b)
    model.eqQinj=Constraint(Bus,rule=eqQinj)


    def eqPbalance(model,b):
        return model.Pinj[b] == sum(model.Pflow[l,i,j] for l,i,j in Lines.select('*',b,'*'))
    model.eqPbalance=Constraint(Bus,rule=eqPbalance)


    def eqQbalance(model,b):
        return model.Qinj[b] == sum(model.Qflow[l,i,j] for l,i,j in Lines.select('*',b,'*'))
    model.eqQbalance=Constraint(Bus,rule=eqQbalance)


    # LP AC equations
    def eqPij_rule(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        g_ij = branch.loc[(l,i,j)]['g_ij']
        b_ij = branch.loc[(l,i,j)]['b_ij']
        return m.Pflow[l,i,j] == 0.5*g_ij*(m.V2[i] - m.V2[j]) - b_ij*(m.theta[i] - m.theta[j])
    model.eqPij = pyo.Constraint(model.Lines, rule=eqPij_rule)

    def eqQij_rule(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        g_ij = branch.loc[(l,i,j)]['g_ij']
        b_ij = branch.loc[(l,i,j)]['b_ij']
        return m.Qflow[l,i,j] == -0.5*b_ij*(m.V2[i] - m.V2[j]) - g_ij*(m.theta[i] - m.theta[j])
    model.eqQij = pyo.Constraint(model.Lines, rule=eqQij_rule)

    def zero_Pflow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Pflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_Pflow_eq = pyo.Constraint(model.Lines, rule=zero_Pflow_eq)

    def zero_Qflow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Qflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_Qflow_eq = pyo.Constraint(model.Lines, rule=zero_Qflow_eq)

    def eqVmax(model, i):
        return model.V2[i] <= vmax**2
    model.eqVmax = pyo.Constraint(Bus, rule=eqVmax)

    def eqVmin(model, i):
        return vmin**2 <= model.V2[i]
    model.eqVmin = pyo.Constraint(Bus, rule=eqVmin)


    # Branch flow limits (P-Q combined)
    beta = 0
    def eqPij1(model, l, i, j):
        return model.Pflow[l,i,j] + np.tan(np.pi/6)*model.Qflow[l,i,j] <= (1-beta)*branch.loc[(l,i,j)]['limit']
    model.eqPij1 = pyo.Constraint(Lines, rule=eqPij1)

    def eqPij2(model, l, i, j):
        return model.Pflow[l,i,j] - np.tan(np.pi/6)*model.Qflow[l,i,j] <= (1-beta)*branch.loc[(l,i,j)]['limit']
    model.eqPij2 = pyo.Constraint(Lines, rule=eqPij2)

    def eqPij3(model, l, i, j):
        return model.Pflow[l,i,j] + np.tan(np.pi/6)*model.Qflow[l,i,j] >= (beta-1)*branch.loc[(l,i,j)]['limit']
    model.eqPij3 = pyo.Constraint(Lines, rule=eqPij3)

    def eqPij4(model, l, i, j):
        return model.Pflow[l,i,j] - np.tan(np.pi/6)*model.Qflow[l,i,j] >= (beta-1)*branch.loc[(l,i,j)]['limit']
    model.eqPij4 = pyo.Constraint(Lines, rule=eqPij4)

    
    def eqtheta(model):
        return model.theta[Bus[0]]==0
    model.eqtheta=Constraint(rule=eqtheta)

    # gen 
    def eqPgmax(model, g): 
        return model.Pg[g] <= Gen_data.loc[g]['Pmax']
    model.eqPgmax = Constraint(G, rule=eqPgmax)

    def eqPgmin(model, g): 
        return Gen_data.loc[g]['Pmin'] <= model.Pg[g]
    model.eqPgmin = Constraint(G, rule=eqPgmin)

    def eqQgmax(model, g): 
        return model.Qg[g] <= Gen_data.loc[g]['Qmax']
    model.eqQgmax = Constraint(G, rule=eqQgmax)

    def eqQgmin(model, g): 
        return Gen_data.loc[g]['Qmin'] <= model.Qg[g]
    model.eqQgmin = Constraint(G, rule=eqQgmin)


    def quad_cost_rule(model):
        return sum( Gen_data.loc[g]['a']*model.Pg[g]**2 + model.gen_cost_b[g]*model.Pg[g]
                   +0.001*Gen_data.loc[g]['a']*model.Qg[g]**2  for g in model.G)

    # === Objective
    model.obj = pyo.Objective(rule=quad_cost_rule, sense=pyo.minimize)

    
    return model        
    
    

# %%


# %%
def Solve_AC_OPF_Pyomo(data,
                    modelX,
                    Pd_instance=None,
                    Qd_instance=None,
                    lines_outage=[],
                    limit_new=None,
                    gen_cost_new=None,
                    cong_limit=0.95, 
                    print_result=False):

    """
    Solve the linearized AC OPF model for a specific operating scenario.

        The function updates demand, line limits, generator costs, and optional line
        outages, solves the AC OPF, and returns objective value, nodal injections,
        congestion indicators, and execution time.

        Parameters
        ----------
        data : dict
            Input network and system data dictionary.
        modelX : pyomo.environ.ConcreteModel
            Base AC-OPF model to clone and solve.
        Pd_instance : array-like, optional
            Active demand vector.
        Qd_instance : array-like, optional
            Reactive demand vector.
        lines_outage : list
            Line IDs removed from service.
        limit_new : array-like, optional
            Updated one-directional line limits.
        gen_cost_new : array-like, optional
            Updated linear generator-cost coefficients.
        cong_limit : float
            Loading-ratio threshold for congestion detection.
        print_result : bool
            If True, print generation cost.

        Returns
        -------
        dict or str
            AC-OPF results, or "Infeasible" if the model fails.
    """
    Bus = data['Bus']
    G = data['G']
    line = data['line']
    Lines1D = data['Lines1D']
    
    model = modelX.clone()
    
    if Pd_instance is None:
        print("DemandInstance is required")
        # Pd_instance = data['Pdemand']['Pd'].to_numpy()
        
    # === Update Pd param
    for d_ind, d in enumerate(data['Demandset']):
        model.Pd[d] = Pd_instance[d_ind]
        model.Qd[d] = Qd_instance[d_ind]

    if limit_new is not None:
        for l_ind,(l,i,j) in enumerate(Lines1D):
            model.limit[l,i,j] = limit_new[l_ind]
            model.limit[l,j,i] = limit_new[l_ind] #bi-directional lines

    if gen_cost_new is not None:
        for g_ind,g in enumerate(G):
            model.gen_cost_b[g] = gen_cost_new[g_ind]


    # === Update outage set and reconstruct flow constraint
    if lines_outage!=[]:
        model.outage_lines.clear()
        model.outage_lines.update(lines_outage)

        model.flow_eq.clear()
        model.flow_eq._constructed = False
        model.flow_eq.construct()

        model.zero_flow_eq.clear()
        model.zero_flow_eq._constructed = False
        model.zero_flow_eq.construct()

    
    # === Solve
    try:
        solver = pyo.SolverFactory("gurobi")
        start_time = time.time()
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        result = SolverFactory("gurobi").solve(model, tee=False, options={"OutputFlag": 0})
        end_time = time.time()
        ex_time = end_time - start_time

        # === Check feasibility
        if (result.solver.termination_condition != pyo.TerminationCondition.optimal) or (result.solver.status != SolverStatus.ok):
            return "Infeasible"
        
        
        if limit_new is not None:
                limit = limit_new.copy()
        else:
            limit = data['branch']['limit'].to_numpy()[:len(data['line'])]    
            
        flows = np.zeros(len(line))
        # Pflow_df = pd.DataFrame(columns=['flow'], index=data['branch1D'].index) 


        for l_ind, (l, i, j) in enumerate(Lines1D):
            flows[l_ind] = np.sqrt(model.Pflow[l, i, j].value**2 + model.Qflow[l, i, j].value**2)
            # Pflow_df.loc[(l,i,j),'flow'] = flows[l_ind]
            

        loading_ratio = flows / limit
        overloaded_mask = np.abs(loading_ratio) >= cong_limit
        Num_cong = np.sum(overloaded_mask)


        # === Nodal injections
        Pinj = np.zeros((len(Bus)))
        Qinj = np.zeros((len(Bus)))

        for b_ind, b in enumerate(Bus):
            Pinj[b_ind] = model.Pinj[b].value
            Qinj[b_ind] = model.Qinj[b].value


        # === Print if needed
        if print_result:
            print(f"\t GenCost: {pyo.value(model.obj):.3f}")
            

        return {
            'GenCost': pyo.value(model.obj),
            'time': ex_time,
            'Pinj': Pinj,
            'Qinj': Qinj,
            'Num_cong': Num_cong,
            'overloaded_mask': overloaded_mask
        }
    except:
        return "Infeasible"





# %% 
# # AC OSS ML  
def Create_ML_Cong_AC_OSS_Pyomo(data,
                                Substation_nodes=None,
                                Max_Sw_bus=0,
                                LineLimit=1.0, # you can relax the line limit to allow force-switching
                                Zinitial=None,
                                Zfixdict=None):
    """
    Create an ML-assisted congestion-management AC OSS Pyomo model.

        This model uses externally supplied active/reactive nodal injections and
        optimizes busbar-splitting decisions to reduce AC congestion. It includes
        linearized AC power-flow equations, busbar assignment constraints, voltage
        coupling constraints, apparent-flow congestion costs, and infeasibility
        slack penalties.

        Parameters
        ----------
        data : dict
            Input network and system data dictionary.
        Substation_nodes : list, optional
            Buses modeled as switchable substations. If None, all buses are modeled
            as substations.
        Max_Sw_bus : int
            Maximum number of substations allowed to split.
        LineLimit : float
            Scaling factor applied to branch limits.
        Zinitial : dict, optional
            Initial values for binary switching variables.
        Zfixdict : dict, optional
            Dictionary for fixing switching configurations.

        Returns
        -------
        model : pyomo.environ.ConcreteModel
            ML-assisted AC-OSS Pyomo model.
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
    MaxV2=data['MaxV2']
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
    # Bus_sub = Substation_nodes #if Substation_nodes else Bus
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

    Qinj_init = {b: 0 for b in Bus}
    model.Q_inj = pyo.Param(model.Bus, initialize=Qinj_init, mutable=True)
    
    limit_init = {(l,i,j): branch.loc[(l, i, j), 'limit'] for l, i, j in Lines}
    model.limit = pyo.Param(model.Lines, initialize=limit_init, mutable=True)

    cong_cost_init = {(l,sg): Cong_sg_data.loc[(l, sg), 's']  for l,sg in model.LineSeg  }
    model.cong_cost_s = pyo.Param(model.LineSeg, initialize=cong_cost_init, mutable=True)

    cong_df_init = {(l,sg): Cong_sg_data.loc[(l, sg), 'df']  for l,sg in model.LineSeg}
    model.cong_df = pyo.Param(model.LineSeg, initialize=cong_df_init, mutable=True)
    
    vmin=0.9; vmax=1.1
    

    # === Variables
    
    model.Pflow = pyo.Var(model.Lines)
    model.Pflow_li = pyo.Var(model.Lines_sub, model.busbar)
    model.Qflow = pyo.Var(model.Lines)
    model.Qflow_li = pyo.Var(model.Lines_sub, model.busbar)
    model.Sflow = pyo.Var(model.Lines)
    model.delta_bi = pyo.Var(model.Bus_sub, model.busbar)
    model.delta_li = pyo.Var(model.Lines)
    model.V2_bi = pyo.Var(model.Bus_sub, model.busbar)
    model.V2_li = pyo.Var(model.Lines)

    model.z_bus = pyo.Var(model.Bus_sub, domain=pyo.Binary)
    model.z_li = pyo.Var(model.Lines_sub, domain=pyo.Binary)
    
    model.spp = pyo.Var(model.Bus_non_sub, domain=pyo.NonNegativeReals)
    model.spn = pyo.Var(model.Bus_non_sub, domain=pyo.NonNegativeReals)
    model.sqp = pyo.Var(model.Bus_non_sub, domain=pyo.NonNegativeReals)
    model.sqn = pyo.Var(model.Bus_non_sub, domain=pyo.NonNegativeReals)
    
    model.spp_bi = pyo.Var(model.Bus_sub, model.busbar, domain=pyo.NonNegativeReals)
    model.spn_bi = pyo.Var(model.Bus_sub, model.busbar, domain=pyo.NonNegativeReals)
    model.sqp_bi = pyo.Var(model.Bus_sub, model.busbar, domain=pyo.NonNegativeReals)
    model.sqn_bi = pyo.Var(model.Bus_sub, model.busbar, domain=pyo.NonNegativeReals)

    model.InfCost = pyo.Var(domain=pyo.NonNegativeReals)

    model.Sflow_ps = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.Sflow_ng = pyo.Var(model.LineSeg, domain=pyo.NonNegativeReals)
    model.CongCost = pyo.Var(model.line, domain=pyo.NonNegativeReals)
    model.TotalCongCost = pyo.Var(domain=pyo.NonNegativeReals)
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


    # AC equations
    # === Balance constraint for substation buses
    def eq_balance_sub_rule(m, b, mbar):
        inj_sum = m.P_inj[b] if mbar == 'busbar1' else 0
        flow_sum = sum(m.Pflow_li[l, i, j, mbar] for l, i, j in Lines.select('*',b,'*'))
        return inj_sum + m.spp_bi[b,mbar] - m.spn_bi[b,mbar] == flow_sum
    model.eq_balance_sub = pyo.Constraint(model.Bus_sub, model.busbar, rule=eq_balance_sub_rule)

    # === Balance constraint for non-substation buses
    def eq_balance_nonsub_rule(m, b):
        inj_sum = m.P_inj[b] 
        flow_sum = sum(m.Pflow[l, i, j] for l, i, j in Lines.select('*',b,'*'))
        return inj_sum + m.spp[b] - m.spn[b] ==  flow_sum
    model.eq_balance_nonsub = pyo.Constraint(model.Bus_non_sub, rule=eq_balance_nonsub_rule)

    def eq_qbalance_sub_rule(m, b, mbar):
        qinj_sum = m.Q_inj[b] if mbar == 'busbar1' else 0
        qflow_sum = sum(m.Qflow_li[l, i, j, mbar] for l, i, j in Lines.select('*',b,'*'))
        return qinj_sum + m.sqp_bi[b,mbar] - m.sqn_bi[b,mbar] == qflow_sum
    model.eq_qbalance_sub_rule = pyo.Constraint(model.Bus_sub, model.busbar, rule=eq_qbalance_sub_rule)

    # === Balance constraint for non-substation buses
    def eq_qbalance_nonsub_rule(m, b):
        qinj_sum = m.Q_inj[b] 
        qflow_sum = sum(m.Qflow[l, i, j] for l, i, j in Lines.select('*',b,'*'))
        return qinj_sum + m.sqp[b] - m.sqn[b] ==  qflow_sum
    model.eq_qbalance_nonsub_rule = pyo.Constraint(model.Bus_non_sub, rule=eq_qbalance_nonsub_rule)

    #AC P-V equations
    def eqPij_rule(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        g_ij = branch.loc[(l,i,j)]['g_ij']
        b_ij = branch.loc[(l,i,j)]['b_ij']
        return m.Pflow[l,i,j] == 0.5*g_ij*(m.V2_li[l,i,j] - m.V2_li[l,j,i]) - b_ij*(m.delta_li[l,i,j] - m.delta_li[l,j,i])
    model.eqPij = pyo.Constraint(model.Lines, rule=eqPij_rule)

    def eqQij_rule(m, l, i, j):
        if l in m.outage_lines:
            return pyo.Constraint.Skip
        g_ij = branch.loc[(l,i,j)]['g_ij']
        b_ij = branch.loc[(l,i,j)]['b_ij']
        return m.Qflow[l,i,j] == -0.5*b_ij*(m.V2_li[l,i,j] - m.V2_li[l,j,i]) - g_ij*(m.delta_li[l,i,j] - m.delta_li[l,j,i])
    model.eqQij = pyo.Constraint(model.Lines, rule=eqQij_rule)


    def zero_Pflow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Pflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_Pflow_eq = pyo.Constraint(model.Lines, rule=zero_Pflow_eq)

    def zero_Qflow_eq(m, l, i, j):
        if l in m.outage_lines:
            return m.Qflow[l, i, j] == 0
        return pyo.Constraint.Skip  # do not add anything for lines in service
    model.zero_Qflow_eq = pyo.Constraint(model.Lines, rule=zero_Qflow_eq)


    def eqVmax(model, l,i,j):
        return model.V2_li[l,i,j] <= vmax**2
    model.eqVmax = pyo.Constraint(Lines, rule=eqVmax)

    def eqVmin(model, l,i,j):
        return vmin**2 <= model.V2_li[l,i,j]
    model.eqVmin = pyo.Constraint(Lines, rule=eqVmin)

    def eqVmax_bi(model, b, i):
        return model.V2_bi[b, i] <= vmax**2
    model.eqVmax_bi = pyo.Constraint(model.Bus_sub, model.busbar, rule=eqVmax_bi)

    def eqVmin_bi(model, b, i):
        return vmin**2 <= model.V2_bi[b, i]
    model.eqVmin_bi = pyo.Constraint(model.Bus_sub, model.busbar, rule=eqVmin_bi)


    # Branch flow limits (P-Q combined)
    beta = 0
    def eqPij1(model, l, i, j):
        return model.Pflow[l,i,j] + np.tan(np.pi/6)*model.Qflow[l,i,j] <= (1-beta)*model.Sflow[l,i,j]
    model.eqPij1 = pyo.Constraint(Lines, rule=eqPij1)

    def eqPij2(model, l, i, j):
        return model.Pflow[l,i,j] - np.tan(np.pi/6)*model.Qflow[l,i,j] <= (1-beta)*model.Sflow[l,i,j]
    model.eqPij2 = pyo.Constraint(Lines, rule=eqPij2)

    def eqPij3(model, l, i, j):
        return model.Pflow[l,i,j] + np.tan(np.pi/6)*model.Qflow[l,i,j] >= (beta-1)*model.Sflow[l,i,j]
    model.eqPij3 = pyo.Constraint(Lines, rule=eqPij3)

    def eqPij4(model, l, i, j):
        return model.Pflow[l,i,j] - np.tan(np.pi/6)*model.Qflow[l,i,j] >= (beta-1)*model.Sflow[l,i,j]
    model.eqPij4 = pyo.Constraint(Lines, rule=eqPij4)
    

    def eqSmin(model, l, i, j):
        return  -branch.loc[(l,i,j)]['limit'] <= model.Sflow[l,i,j]
    model.eqSmin = pyo.Constraint(Lines, rule=eqSmin)

    def eqSmax(model, l, i, j):
        return  model.Sflow[l,i,j] <= branch.loc[(l,i,j)]['limit'] 
    model.eqSmax = pyo.Constraint(Lines, rule=eqSmax)


    l0, i0, j0 = Lines[0]
    model.ref_bus_angle = pyo.Constraint(expr=model.delta_li[l0, i0, j0] == 0)



    # ---------------- P flow constraints ----------------
    def eq_Pflow1min_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit <= m.Pflow_li[l, i, j, 'busbar1']
    model.eq_Pflow1min = pyo.Constraint(model.Lines_sub, rule=eq_Pflow1min_rule)

    def eq_Pflow1max_rule(m, l, i, j):
        return m.Pflow_li[l, i, j, 'busbar1'] <= (1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit
    model.eq_Pflow1max = pyo.Constraint(model.Lines_sub, rule=eq_Pflow1max_rule)

    def eq_Pflow2min_rule(m, l, i, j):
        return -m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit <= m.Pflow_li[l, i, j, 'busbar2']
    model.eq_Pflow2min = pyo.Constraint(model.Lines_sub, rule=eq_Pflow2min_rule)

    def eq_Pflow2max_rule(m, l, i, j):
        return m.Pflow_li[l, i, j, 'busbar2'] <= m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit
    model.eq_Pflow2max = pyo.Constraint(model.Lines_sub, rule=eq_Pflow2max_rule)

    def eq_Pflow_sum_rule(m, l, i, j):
        return m.Pflow[l, i, j] == m.Pflow_li[l, i, j, 'busbar1'] + m.Pflow_li[l, i, j, 'busbar2']
    model.eq_Pflow = pyo.Constraint(model.Lines_sub, rule=eq_Pflow_sum_rule)


    # ---------------- Q flow constraints ----------------
    def eq_Qflow1min_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit <= m.Qflow_li[l, i, j, 'busbar1']
    model.eq_Qflow1min = pyo.Constraint(model.Lines_sub, rule=eq_Qflow1min_rule)

    def eq_Qflow1max_rule(m, l, i, j):
        return m.Qflow_li[l, i, j, 'busbar1'] <= (1 - m.z_li[l, i, j]) * m.limit[l, i, j] * LineLimit
    model.eq_Qflow1max = pyo.Constraint(model.Lines_sub, rule=eq_Qflow1max_rule)

    def eq_Qflow2min_rule(m, l, i, j):
        return -m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit <= m.Qflow_li[l, i, j, 'busbar2']
    model.eq_Qflow2min = pyo.Constraint(model.Lines_sub, rule=eq_Qflow2min_rule)

    def eq_Qflow2max_rule(m, l, i, j):
        return m.Qflow_li[l, i, j, 'busbar2'] <= m.z_li[l, i, j] * m.limit[l, i, j] * LineLimit
    model.eq_Qflow2max = pyo.Constraint(model.Lines_sub, rule=eq_Qflow2max_rule)

    def eq_Qflow_sum_rule(m, l, i, j):
        return m.Qflow[l, i, j] == m.Qflow_li[l, i, j, 'busbar1'] + m.Qflow_li[l, i, j, 'busbar2']
    model.eq_Qflow = pyo.Constraint(model.Lines_sub, rule=eq_Qflow_sum_rule)



    #delta 
    def eq_delta_bus1_rule(m, b):
        return -BigM_b * (1 - m.z_bus[b]) <= m.delta_bi[b, 'busbar1'] - m.delta_bi[b, 'busbar2']
    model.eq_delta_bus1 = pyo.Constraint(model.Bus_sub, rule=eq_delta_bus1_rule)

    def eq_delta_bus2_rule(m, b):
        return m.delta_bi[b, 'busbar1'] - m.delta_bi[b, 'busbar2'] <= BigM_b * (1 - m.z_bus[b])
    model.eq_delta_bus2 = pyo.Constraint(model.Bus_sub, rule=eq_delta_bus2_rule)


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
            

    
    #V2
    def eq_Vdiff1_bus_rule(m, b):
        return -MaxV2 * (1 - m.z_bus[b]) <= m.V2_bi[b, 'busbar1'] - m.V2_bi[b, 'busbar2']
    model.eq_Vdiff1_bus = pyo.Constraint(model.Bus_sub, rule=eq_Vdiff1_bus_rule)

    def eq_Vdiff2_bus_rule(m, b):
        return m.V2_bi[b, 'busbar1'] - m.V2_bi[b, 'busbar2'] <= MaxV2 * (1 - m.z_bus[b])
    model.eq_Vdiff2_bus = pyo.Constraint(model.Bus_sub, rule=eq_Vdiff2_bus_rule)


    def eq_V2_lb1Frmin_rule(m, l, i, j):
        return -m.z_li[l, i, j] * MaxV2 <= m.V2_li[l, i, j] - m.V2_bi[i, 'busbar1']
    model.eq_V2_lb1Frmin = pyo.Constraint(model.Lines_sub, rule=eq_V2_lb1Frmin_rule)

    def eq_V2_lb1Frmax_rule(m, l, i, j):
        return m.V2_li[l, i, j] - m.V2_bi[i, 'busbar1'] <= m.z_li[l, i, j] * MaxV2
    model.eq_V2_lb1Frmax = pyo.Constraint(model.Lines_sub, rule=eq_V2_lb1Frmax_rule)


    def eq_V2_lb2Frmin_rule(m, l, i, j):
        return -(1 - m.z_li[l, i, j]) * MaxV2 <= m.V2_li[l, i, j] - m.V2_bi[i, 'busbar2']
    model.eq_V2_lb2Frmin = pyo.Constraint(model.Lines_sub, rule=eq_V2_lb2Frmin_rule)

    def eq_V2_lb2Frmax_rule(m, l, i, j):
        return m.V2_li[l, i, j] - m.V2_bi[i, 'busbar2'] <= (1 - m.z_li[l, i, j]) * MaxV2
    model.eq_V2_lb2Frmax = pyo.Constraint(model.Lines_sub, rule=eq_V2_lb2Frmax_rule)


    model.eqV2 = pyo.ConstraintList()
    for b in Bus_non_sub:
        lmin, b, j1 = Lines.select('*', b, '*')[0]  # pick first line at bus b
        for l, i, j in Lines.select('*', b, '*'):
            model.eqV2.add(
                model.V2_li[lmin, b, j1] == model.V2_li[l, i, j]
            )


   
    # === Congestion power flow decomposition
    def eqPflowSumCong_rule(m, l, i, j):
        return m.Sflow[l, i, j] == sum(m.Sflow_ps[l, sg] for sg in m.Seg) - sum(m.Sflow_ng[l, sg] for sg in m.Seg)
    model.eqPflowSumCong = pyo.Constraint(model.Lines1D, rule=eqPflowSumCong_rule)

    # === Upper bounds on Pflow_ps
    def eqPflow_ps_SgMax_rule(m, l, sg):
        return m.Sflow_ps[l, sg] <= m.cong_df[l,sg]
    model.eqPflow_ps_SgMax = pyo.Constraint(model.line, model.Seg, rule=eqPflow_ps_SgMax_rule)

    # === Upper bounds on Pflow_ng
    def eqPflow_ng_SgMax_rule(m, l, sg):
        return m.Sflow_ng[l, sg] <= m.cong_df[l,sg]  
    model.eqPflow_ng_SgMax = pyo.Constraint(model.line, model.Seg, rule=eqPflow_ng_SgMax_rule)

    # === Congestion cost per line
    def eqCongCost_rule(m, l, i, j):
        return m.CongCost[l] == sum(m.Sflow_ps[l, sg] * m.cong_cost_s[l,sg] for sg in m.Seg) + \
                                sum(m.Sflow_ng[l, sg] * m.cong_cost_s[l,sg] for sg in m.Seg)
    model.eqCongCost = pyo.Constraint(model.Lines1D, rule=eqCongCost_rule)

    # === Total congestion cost
    model.eqTotalCongCost = pyo.Constraint(expr=model.TotalCongCost == sum(model.CongCost[l] for l in model.line))

    # === Link total cost to objective
    model.Eq_OF = pyo.Constraint(expr=model.OF == model.TotalCongCost + model.InfCost)   
    
    model.Eq_Inf = pyo.Constraint(expr=model.InfCost == 1e+4*sum(model.spp[b] + model.spn[b] + model.sqp[b] + model.sqn[b]     for b in model.Bus_non_sub) 
                                + 1e+4*sum(model.spp_bi[b,i] + model.spn_bi[b,i] + model.sqp_bi[b,i] + model.sqn_bi[b,i] for b in model.Bus_sub for i in model.busbar ) )   

    # === Objective function
    model.obj = pyo.Objective(expr=model.OF, sense=pyo.minimize)



    return model









# %%
def Solve_ML_Cong_AC_OSS_Pyomo(data,
                               modelX,
                               Substation_nodes=None,
                               lines_outage=[],
                               P_inj=None,
                               Q_inj=None,
                               limit_new=None,
                               candidate_bus=None,
                               force_split=[],
                               cong_limit=0.95,
                               print_result=False):
    """
    Solve the ML-assisted congestion-management AC OSS model.

        The function clones a base AC-OSS model, updates active/reactive injections,
        line limits, congestion-cost segments, candidate switching constraints, and
        forced splitting decisions. It then solves the MIP and returns line flows,
        congestion status, selected split substations, congestion cost, and runtime.

        Parameters
        ----------
        data : dict
            Input network and system data dictionary.
        modelX : pyomo.environ.ConcreteModel
            Base ML-assisted AC-OSS model to clone and solve.
        Substation_nodes : list, optional
            Buses modeled as switchable substations.
        lines_outage : list
            Line IDs removed from service.
        P_inj : array-like, optional
            Active nodal injection vector.
        Q_inj : array-like, optional
            Reactive nodal injection vector.
        limit_new : array-like, optional
            Updated one-directional line limits.
        candidate_bus : list, optional
            Candidate substations allowed to split.
        force_split : list
            Substations forced to split.
        cong_limit : float
            Loading-ratio threshold for congestion detection.
        print_result : bool
            If True, print congestion and switching results.

        Returns
        -------
        dict or str
            AC-OSS results, or "Infeasible" if the model fails.
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
    
    if P_inj is not None:
        for b_ind, b in enumerate(data['Bus']):
            model.P_inj[b] = P_inj[b_ind]
    if Q_inj is not None:
        for b_ind, b in enumerate(data['Bus']):
            model.Q_inj[b] = Q_inj[b_ind]
    

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
                # for g, i in G2B.select('*',b):
                #     model.fix_zg.add(model.z_g[g] == 0)
                # for d, i in D2B.select('*',b):
                #     model.fix_zd.add(model.z_d[d] == 0)
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
        result = SolverFactory("gurobi").solve(model, tee=True, options={"OutputFlag": 1,'TimeLimit': 3600})
        end_time = time.time()
        ex_time = end_time - start_time

        # === Check feasibility
        if result.solver.termination_condition in [TerminationCondition.infeasible, TerminationCondition.unbounded]:
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
        Pflows = np.zeros(len(line))
        Qflows = np.zeros(len(line))
        Sflows = np.zeros(len(line))
        for l_ind, (l, i, j) in enumerate(Lines1D):
            Pflows[l_ind] = model.Pflow[l, i, j].value
            Qflows[l_ind] = model.Qflow[l, i, j].value
            Sflows[l_ind] = model.Sflow[l, i, j].value

        loading_ratio = Sflows / limit
        overloaded_mask = np.abs(loading_ratio) >= cong_limit
        Num_cong = np.sum(overloaded_mask)


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
            'Pflow': Pflows,
            'Qflow': Qflows,
            'Sflow': Sflows,
            # 'Pflow_top': Pflow_top,
            'Num_cong': Num_cong,
            'z_bus_open':z_bus_open,
            'CongCost':model.TotalCongCost.value
        }
    except:
        return "Infeasible"


    