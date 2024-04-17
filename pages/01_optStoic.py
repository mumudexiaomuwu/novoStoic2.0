import streamlit as st
import pulp
import pandas as pd
import numpy as np
import re
from PIL import Image
import webbrowser
import json
import pickle
import sys 
import joblib
import ast
from math import gcd
import math
import os
sys.path.append('./../data/CC/')

import chemaxon
from chemaxon import *
from compound import Compound
from compound_cacher import CompoundCacher
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

def find_factor(numbers):
    # Function to find the least common multiple (LCM)
    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b) if a and b else 0
    
    # Find the LCM of all denominators
    lcm_denominator = 1
    for num in numbers:
        lcm_denominator = lcm(lcm_denominator, int(num * (10**len(str(num)))))
    
    return lcm_denominator

def convert_to_integer(numbers):
    factor = find_factor(numbers)
    #print(factor)
    return [int(num * (factor / den)) for num, den in zip(numbers, [1] * len(numbers))], factor


def extract_det(smiles):
    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    charge = smiles.count('+')-smiles.count('-')
    elem = ['Cl','Mg','Fe','Se','Co','As','Br','C','H','N','O','P','S','F','I','R']
    if '+' in formula:
        formula = formula.split('+')[0]
    if '-' in formula:
        formula = formula.split('-')[0]
    details = {}
    for i in range(len(elem)):
        
        if elem[i] in formula:
            ind = formula.find(elem[i])
            remove_ind = []
            remove_ind.append(ind)
            if i<=6:
                ind+=2
            else:
                ind+=1
            if ind>len(formula)-1:
                remove_ind.append(ind)
                val = 1
            else:
                count = ''
                #print(formula)
                #print("----------")
                while formula[ind].isalpha() is False:
                    count = count+formula[ind]
                    ind+=1
                    if ind>len(formula)-1:
                        break
                    #remove_ind.append(ind)
                remove_ind.append(ind)
                if count=='':
                    val = 1
                else:
                    val = int(count)
            formula = formula[:remove_ind[0]]+formula[remove_ind[1]:]
            
        else:
            val = 0
        details[elem[i]]=val
        #print("------------")
    details['Charge']=int(charge)
    details_ordered = {'C': 0,
 'H': 1,
 'N': 0,
 'O': 4,
 'P': 1,
 'S': 0,
 'F': 0,
 'Cl': 0,
 'Mg': 0,
 'Fe': 0,
 'Se': 0,
 'Co': 0,
 'As': 0,
 'Br': 0,
 'I': 0,
 'R': 0,
 'Charge': -2}

    for x in details_ordered:
        details_ordered[x] = details[x]

    return details_ordered

@st.cache_data
def load_smiles():
    db = pd.read_csv('./../data/cache_compounds_20160818.csv',
                     index_col='compound_id')
    db_smiles = db['smiles_pH7'].to_dict()
    db_inchi = db_inchi = db['inchi'].to_dict()
    db_atom_bag = db['atom_bag'].to_dict()
    db_hydrogen = {}
    for i in db_atom_bag:
        db_hydrogen[i]=kegg_hydrogen(db_atom_bag[i])
    
    return db_smiles,db_inchi,db_hydrogen


@st.cache_data
def load_molsig_rad1():
    molecular_signature_r1 = json.load(open('./../data/decompose_vector_ac.json'))
    return molecular_signature_r1

@st.cache_data
def load_molsig_rad2():
    molecular_signature_r2 = json.load(
        open('./../data/decompose_vector_ac_r2_py3_indent_modified_manual.json'))
    return molecular_signature_r2

@st.cache_data
def load_model():
    filename = './../data/models/dGPredictor/M12_model_BR.pkl'
    loaded_model = joblib.load(open(filename, 'rb'))
    return loaded_model

@st.cache_data
def load_compound_cache():
    ccache = CompoundCacher()
    return ccache

@st.cache_data
def load_metab_df():
    metab_df = pd.read_csv("./../data/optStoic/metanetx_metab_db_noduplicates.csv" , index_col = "Unnamed: 0")
    return metab_df

@st.cache_data
def load_sij_dict():
    sij_dict = json.load(open("./../data/optStoic/metanetx_sij_final.json"))
    return sij_dict

@st.cache_data
def load_metab_detail_dict():
    metab_detail_dict = json.load(open("./../data/optStoic/metab_detail_dict_final.json"))
    return metab_detail_dict

@st.cache_data
def load_met_2_kegg():
    met_2_kegg = json.load(open("./../data/optStoic/met_2_kegg.json"))
    return met_2_kegg

@st.cache_data
def load_kegg_2_met():
    kegg_2_met = json.load(open("./../data/optStoic/kegg_2_met.json"))
    return kegg_2_met

@st.cache_data
def load_allow_moiety_dict():
    allow_moiety_dict = json.load(open("./../data/optStoic/allow_moiety_dict.json"))
    return allow_moiety_dict

def kegg_hydrogen(atom_bag):
    p = atom_bag
    p = p.replace("u","")
    p = ast.literal_eval(p)
    val = 0
    if "H" in p:
        return p['H']
    else:
        return val
    

def count_substructures(radius, molecule):
    """Helper function for get the information of molecular signature of a
    metabolite. The relaxed signature requires the number of each substructure
    to construct a matrix for each molecule.
    Parameters
    ----------
    radius : int
        the radius is bond-distance that defines how many neighbor atoms should
        be considered in a reaction center.
    molecule : Molecule
        a molecule object create by RDkit (e.g. Chem.MolFromInchi(inchi_code)
        or Chem.MolToSmiles(smiles_code))
    Returns
    -------
    dict
        dictionary of molecular signature for a molecule,
        {smiles: molecular_signature}
    """
    m = molecule
    smi_count = dict()
    atomList = [atom for atom in m.GetAtoms()]

    for i in range(len(atomList)):
        env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, i)
        atoms = set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())

        # only one atom is in this environment, such as O in H2O
        if len(atoms) == 0:
            atoms = {i}

        smi = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms),
                                       bondsToUse=env, canonical=True)

        if smi in smi_count:
            smi_count[smi] = smi_count[smi] + 1
        else:
            smi_count[smi] = 1
    return smi_count

def get_rule(rxn_dict, molsig1, molsig2, novel_decomposed1, novel_decomposed2):
    
    if novel_decomposed1 != None:
        for cid in novel_decomposed1:
            molsig1[cid] = novel_decomposed1[cid]
    if novel_decomposed2 != None:
        for cid in novel_decomposed2:
            molsig2[cid] = novel_decomposed2[cid]

    molsigna_df1 = pd.DataFrame.from_dict(molsig1).fillna(0)
    all_mets1 = molsigna_df1.columns.tolist()
    all_mets1.append("C00080")
    all_mets1.append("C00282")

    molsigna_df2 = pd.DataFrame.from_dict(molsig2).fillna(0)
    all_mets2 = molsigna_df2.columns.tolist()
    all_mets2.append("C00080")
    all_mets2.append("C00282")

    moieties_r1 = open('./../data/group_names_r1.txt')
    moieties_r2 = open('./../data/group_names_r2_py3_modified_manual.txt')
    moie_r1 = moieties_r1.read().splitlines()
    moie_r2 = moieties_r2.read().splitlines()

    molsigna_df1 = molsigna_df1.reindex(moie_r1)
    molsigna_df2 = molsigna_df2.reindex(moie_r2)

    rule_df1 = pd.DataFrame(index=molsigna_df1.index)
    rule_df2 = pd.DataFrame(index=molsigna_df2.index)
    # for rid, value in reaction_dict.items():
    #     # skip the reactions with missing metabolites
    #     mets = value.keys()
    #     flag = False
    #     for met in mets:
    #         if met not in all_mets:
    #             flag = True
    #             break
    #     if flag: continue

    rule_df1['change'] = 0
    for met, stoic in rxn_dict.items():
        if met == "C00080" or met == "C00282":
            continue  # hydogen is zero
        rule_df1['change'] += molsigna_df1[met] * stoic

    rule_df2['change'] = 0
    for met, stoic in rxn_dict.items():
        if met == "C00080" or met == "C00282":
            continue  # hydogen is zero
        rule_df2['change'] += molsigna_df2[met] * stoic

    rule_vec1 = rule_df1.to_numpy().T
    rule_vec2 = rule_df2.to_numpy().T

    m1, n1 = rule_vec1.shape
    m2, n2 = rule_vec2.shape

    zeros1 = np.zeros((m1, 44))
    zeros2 = np.zeros((m2, 44))
    X1 = np.concatenate((rule_vec1, zeros1), 1)
    X2 = np.concatenate((rule_vec2, zeros2), 1)

    rule_comb = np.concatenate((X1, X2), 1)

    # rule_df_final = {}
    # rule_df_final['rad1'] = rule_df1
    # rule_df_final['rad2'] = rule_df2
    return rule_comb, rule_df1, rule_df2

def get_alt_mean(loaded_model):
    alt_X = np.zeros([26404,26404])
    for x_ind,x_val in enumerate(alt_X):
        alt_X[x_ind,x_ind]=1
            
    #print("Printing alt_X shape = ",alt_X.shape)
    #print("Printing alt_X")
    #print(alt_X)
    alt_ymean, alt_ystd = loaded_model.predict(alt_X,return_std=True)
    #print("Printing alt_ymean shape = ",alt_ymean.shape)
    #print("Print alt_ymean")
    #print(alt_ymean)
    #print("Printing alt_ystd shape = ",alt_ystd.shape)

    #final_ymean = np.sum(X[0]*alt_ymean)
    #print("Manually calculated mean = ",final_ymean)
    return list(alt_ymean)

    

def optimal_stoic(reactant,product,add_info):
    substrate = reactant # glucose
    pdt = [product] #acetate
    allow = ['WATER','MNXM3','MNXM8','MNXM10','MNXM738702','MNXM5','MNXM735438','MNXM40333','MNXM9','MNXM727276','MNXM13','MNXM1','MNXM1108018','MNXM728294','MNXM11','MNXM729302','MNXM732620']

    #st.write("BEFORE parsing add info => Allow = ", allow)
    #st.write("Len of add_info = ", len(add_info))
    #st.write("add_info = ", add_info)
    #st.write("Substrate = ",substrate)
    #st.write("Product = ",pdt[0])
    if len(add_info)>0:
        for i in add_info:
            #st.write('I = ', i)
            #st.write("add_info_"+i+" = ", add_info[i])
            if i!= substrate and i!=pdt[0]:
                allow.append(i)
    #st.write("BEFORE adding substrate and product => Allow = ", allow)
    metab_df = load_metab_df()
    metab_df.loc["MNXM9"] = ['phosphate', 'HO4P',-2.0,95.96234,'InChI=1S/H3O4P/c1-5(2,3)4/h(H3,1,2,3,4)/p-2',
                                    'InChIKey=NBIIXXVUZAFLBC-UHFFFAOYSA-L', 'O=P([O-])([O-])O', 'chebi:43474']
    metab_df.loc["MNXM1"] = ['H(+)', 'H',1.0,1.00794,'InChI=1S/p+1',
                                    'InChIKey=GPRLSGONYQIRFK-UHFFFAOYSA-N', '[H+]', 'mnx:PROTON']
    separate = ['MNXM3']
    #[NADH, NAD+], [NADPH, NADP+], 
    pairs = [['MNXM10','MNXM8'],['MNXM738702','MNXM5']]
    #[ATP,ADP],[ATP,AMP], [dips,AMP], [phos,ADP]
    cond_pairs = [['MNXM3', 'MNXM40333'],['MNXM3','MNXM728294'], ['MNXM11','MNXM728294'], ['MNXM9','MNXM40333']]
    #['MNXM1','MNXM8'], ['MNXM1','MNXM5']
    #[H+,NAD+], [H+,NADP+]
    #[ATP,ADP,AMP]
    cond_threes = [['MNXM3', 'MNXM40333', 'MNXM728294']]

    
    
    sij_dict = load_sij_dict()
    metab_detail_dict = load_metab_detail_dict()
    #st.write(" metab_detail_dict = ", metab_detail_dict)
    #st.write("type of metab_detail_dict = ", type(metab_detail_dict))
    #st.write("len of metab_detail_dict = ", len(metab_detail_dict))
    met_2_kegg = load_met_2_kegg()
    kegg_2_met = load_kegg_2_met()
    allow_moiety_dict = load_allow_moiety_dict()
    rxns = list(sij_dict.keys())
    mets = list(metab_detail_dict.keys())
    elems = list(metab_detail_dict['MNXM8'].keys())

    db_smiles, db_inchi, db_hydrogen = load_smiles()
    #db_inchi = load_inchi()
    molsig_r1 = load_molsig_rad1()
    molsig_r2 = load_molsig_rad2()

    rid ="random"

    loaded_model = load_model()
    ccache = load_compound_cache()
    pH = 7 # any number between 0-14 
    I = 0.1
    if "C" in substrate:
        substrate = kegg_2_met[substrate]
    if "C" in pdt[0]:
        pdt[0] = kegg_2_met[pdt[0]]
    
    allow.append(substrate)
    allow.append(pdt[0])
    #### Optimization problem starts
    stoi_vars = pulp.LpVariable.dicts("stoichiometry", allow, lowBound=-5, upBound=5, cat='Integer')
    bin_vars = pulp.LpVariable.dicts("active",allow,lowBound=0, upBound=1, cat='Binary')
    lp_prob = pulp.LpProblem("Objective_problem", pulp.LpMaximize)
    lp_prob += stoi_vars[pdt[0]]

    #_____Getting mean dG ousing the stoichiometry variables
    metab_df_smiles = metab_df['SMILES'].to_dict()
    metab_df_inchi = metab_df['InChI'].to_dict()
    metab_df_name = metab_df['Name'].to_dict()

    dG_expression = np.zeros(26404)
    dG_expression = list(dG_expression)
    #rxn_dict, add_info, rid, pH, I, loaded_model, molsig_r1, molsig_r2
    #st.write("After adding substrate and product => Allow = ", allow)
    for id in allow:
        if id not in allow_moiety_dict:
            flag = 0
            smiles = 'nothing'
            if id not in met_2_kegg:
                #st.write(id)
                if 'MNXM' in id:
                    smiles = metab_df_smiles[id]
                else:
                    smiles = add_info[id]
                temp_dict = {id:1}
                flag = 1
            else:
                temp_dict = {met_2_kegg[id]:1}
                if met_2_kegg[id] not in db_smiles:
                    smiles = metab_df_smiles[id]
                    flag = 2
                    #st.write("This Kegg ID & smiles is not in the database = " + met_2_kegg[id])
            if smiles!='nothing':
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.RemoveHs(mol)
                # Chem.RemoveStereochemistry(mol)
                smi_count = count_substructures(1, mol)
                smi_count_2 = count_substructures(2,mol)
                if flag==1:
                    if id not in metab_detail_dict:
                        metab_detail_dict[id] = extract_det(smiles)
                    #metab_detail_dict
                    molsig_r1[id] = smi_count
                    molsig_r2[id] = smi_count_2
                    #st.write(id + " smiles = " + smiles)
                if flag==2:
                    molsig_r1[met_2_kegg[id]] = smi_count
                    molsig_r2[met_2_kegg[id]] = smi_count_2
                    #st.write(met_2_kegg[id] + " smiles = " + smiles)
            
            rule_comb, rule_df1, rule_df2 = get_rule(
                temp_dict, molsig_r1, molsig_r2, [], [])
            rule_comb = rule_comb[0]
            rule_comb = list(rule_comb)
            allow_moiety_dict[id] = rule_comb
        for ind,x in enumerate(dG_expression):
            dG_expression[ind] += stoi_vars[id]*allow_moiety_dict[id][ind]
            
    alt_ymean = get_alt_mean(loaded_model)
    
    dG_sum = sum([a*b for a,b in zip(dG_expression,alt_ymean)])
    lp_prob += dG_sum <= 5.0, "dG_constraint"
    for j in elems:
        lp_prob += pulp.lpSum([stoi_vars[i]*metab_detail_dict[i][j] for i in allow]) == 0
    #Constraint 2 (NADPH, NADP+ should be equal and opposite in no.)
    #lp_prob += stoi_vars['MNXM738702']+stoi_vars['MNXM5'] == 0
    for couple in pairs:
        lp_prob += stoi_vars[couple[0]]+stoi_vars[couple[1]]==0
        lp_prob += bin_vars[couple[0]]-bin_vars[couple[1]]==0
    
    for couple in cond_pairs:
        lp_prob += -10*(1-bin_vars[couple[1]]) <= stoi_vars[couple[0]] + stoi_vars[couple[1]]
        lp_prob += 10*(1-bin_vars[couple[1]]) >= stoi_vars[couple[0]] + stoi_vars[couple[1]]
    
    #lp_prob += -10*(1-bin_vars_var
        
    for threes in cond_threes:
        lp_prob += bin_vars[threes[0]]<= bin_vars[threes[1]]+bin_vars[threes[2]] 
    
    #ADP & AMP should simultaneously not appear
    lp_prob += bin_vars['MNXM728294']+bin_vars['MNXM40333'] <= 1
    
    #NAD+, NADP+ should not simultaneously appear
    lp_prob += bin_vars['MNXM8']+bin_vars['MNXM5'] <= 1
    
    # CO2 should be in the product
    lp_prob += stoi_vars['MNXM13']>=0

    lp_prob += bin_vars['MNXM10']==1

    lp_prob += stoi_vars[pdt[0]]<=1, 'suboptimal'

    lp_prob += bin_vars['MNXM9']==0, 'phosphate'
    
    for i in allow:
        lp_prob += bin_vars[i]*10 >= stoi_vars[i]
        lp_prob += -1*bin_vars[i]*10 <= stoi_vars[i]

    # allow only upto 8 cofactors
    #lp_prob += pulp.lpSum([bin_vars[id] for id in allow])<= 10
    
    #Constaint 4 (the reactant stoichiometry is fixed to 1)
    lp_prob += stoi_vars[substrate] <= -1
    lp_prob += stoi_vars[substrate] >= -10
    pulp_solver = pulp.CPLEX_CMD(path=None,keepFiles=0, mip=1, msg=1)
    #pulp_solver = pulp.CPLEX_CMD(path=None,keepFiles=0, mip=0, msg=1)
    lp_prob.solve(pulp_solver)
    
    itr = 0

    prev_itr=0

    # Define the folder path
    folder_path = './../Results/optStoic_solutions/'+pdt[0]+'/int_cut_ids/''
    print(folder_path)
    print(os.listdir(folder_path))
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        #print(filename)
        # Check if the file ends with '.txt'
        if filename.endswith('.txt'):
            prev_itr+=1
            # Construct the full path of the file
            file_path = os.path.join(folder_path, filename)
            #print('1')
            #print(file_path)
            # Open and read the file
            with open(file_path, 'r') as file:
                # Do whatever you want with the file content
                file_content = file.read()
                file_content_list = file_content.split("\n") 
                file_content_list= file_content_list[:-1]
                #st.write(file_content)
                length = len(file_content_list) - 1
                #total_vals = sum(int_cut_vals) - 1
                lp_prob += (pulp.lpSum([bin_vars[r] for r in file_content_list]) <= length, "integer_cut_" + str(prev_itr))
               
                    
                    
              


    while pulp.LpStatus[lp_prob.status] == 'Optimal':
        st.write("Found optimal")
        if itr>=20:
            break
        
        itr+=1
        int_cut_ids = []
        int_cut_vals = []
        #print("Entered here\n")
        obj = pulp.value(lp_prob.objective)
       
        #print("Started wriri\n")
        soln_dict = {}
        rxn_dict = {}
        for id in allow:
            if stoi_vars[id].varValue != 0:
                #st.write(str(id)+" = "+ str(stoi_vars[id].varValue)+" and "+ str(bin_vars[id].varValue))
                #id = str(v.name).split('_')[1]
                soln_dict[id]=stoi_vars[id].varValue
                if id in met_2_kegg:
                    rxn_dict[met_2_kegg[id]] = soln_dict[id]
                else:
                    rxn_dict[id] = soln_dict[id]
                    
                int_cut_ids.append(id)
                int_cut_vals.append(bin_vars[id].varValue)
                #st.write('\n')


        dG_val_lower, dG_val_upper = get_lower_limit(rxn_dict, add_info, rid, pH, I, loaded_model, molsig_r1, molsig_r2)

        #if dG_val_lower <= 5.0
        st.write("Theoretical yield = {}\n".format(obj))
        
    
        st.write('\n')

        st.write('\n')
        soln_react_dict = {}
        soln_prod_dict = {}


        all_keys = list(soln_dict.keys()).copy()
        for id in all_keys:
            if abs(soln_dict[id])<1E-3:
                del soln_dict[id]
                #del soln_dict_abs[id]
        #numbers = list(soln_dict_abs.values())
        #integer_list, factor = convert_to_integer(numbers)
        #st.write("Factor = ", factor)
       

        
        for id,val in soln_dict.items():
            if float(val)<0:
                soln_react_dict[id]=val
            else:
                soln_prod_dict[id]=val
                    
        to_print_sol = ''
        to_print_sol_int = ''
        
        
        for id,val in soln_react_dict.items():
            if "MNXM" in id or "WATER" in id:
                to_print_sol+=str(round(val,2))[1:]+' '+metab_df_name[id]
                to_print_sol+=' '+'+'+' '  
                #to_print_sol_int+=soln_react_dict_int[id]+' '+metab_df_name[id]
                #to_print_sol_int+=' '+'+'+' '
            else:
                st.write("react")
                #st.write("Generate molecular formula from this novel molecule")
                to_print_sol+=str(round(val,2))[1:]+' '+id
                to_print_sol+=' '+'+'+' '
                #to_print_sol_int+=soln_react_dict_int[id]+' '+id
                #to_print_sol_int+=' '+'+'+' '
                
        to_print_sol = to_print_sol[:-2]    
        to_print_sol+=' '+'----->'+' '
        #to_print_sol_int+=' '+'----->'+' '
        for id,val in soln_prod_dict.items():
            if "MNXM" in id or "WATER" in id:
                to_print_sol+=str(round(val,2))+' '+metab_df_name[id]
                to_print_sol+=' '+'+'+' '
                #to_print_sol_int+=soln_prod_dict_int[id]+' '+metab_df_name[id]
                #to_print_sol_int+=' '+'+'+' '

            else:
                #st.write(id)
                #st.write("Generate molecular formula from this novel molecule")
                to_print_sol+=str(round(val,2))+' '+id
                to_print_sol+=' '+'+'+' '
                #to_print_sol_int+=soln_prod_dict_int[id]+' '+id
                #to_print_sol_int+=' '+'+'+' '
        #st.write("Found LCM of this list = ", a)
        #st.write("LCM = ", lcm)
       
        with st.container(border=True):
            #st.write("Actual hydrogen = ", temp_val_float)
            st.markdown(to_print_sol[:-2])


       
        #st.write(to_print_sol)       
        st.write('\n---------------------------------------------\n')
    
        length = len(int_cut_ids) - 1
        total_vals = sum(int_cut_vals) - 1
        #total_vals = sum([int(soln_dict[id]) for id in int_cut_ids])
        #total_vals_up = total_vals+1
        #total_vals_down = total_vals-1
        #lp_prob += (pulp.lpSum([stoi_vars[r] for r in int_cut_ids]) <= total_vals_down,"integer_cut_down" + str(itr+prev_itr))
    
        #lp_prob += (pulp.lpSum([stoi_vars[r] for r in int_cut_ids]) >= total_vals_up,  "integer_cut_up" + str(itr+prev_itr))
        lp_prob += (pulp.lpSum([bin_vars[r] for r in int_cut_ids]) <= total_vals,  "integer_cut" + str(itr+prev_itr))

        with open('./../Results/optStoic_solutions/'+pdt[0]+'/int_cut_ids/'+str(itr+prev_itr)+'.txt', 'w') as f:
            for line in int_cut_ids:
                f.write(f"{line}\n")
    
        #eps = 0.02

        soln_dict_common_name = {}
        soln_dict_metanetx_id = {}
        for id,val in soln_dict.items():
            if "MNXM" in id or "WATER" in id:
                #to_print_sol+=val+' '+metab_df_name[id]
                soln_dict_common_name[metab_df_name[id]] = round(val)
            else:
                soln_dict_common_name[id]=round(val)

        for id,val in soln_dict.items():
            soln_dict_metanetx_id[id]=round(val)
        
        soln_dict_common_name['dG_lower'] = dG_val_lower
        soln_dict_common_name['dG_upper'] = dG_val_upper
        
        lp_prob.solve(pulp_solver)
        with open('./../Results/optStoic_solutions/'+pdt[0]+'/common_name/soln_dict_common_name'+str(itr+prev_itr)+'.json', "w") as outfile: 
            json.dump(soln_dict_common_name, outfile, indent=4)

        with open('./../Results/optStoic_solutions/'+pdt[0]+'/metanetx_id/soln_dict_metanetx_id'+str(itr+prev_itr)+'.json', "w") as outfile: 
            json.dump(soln_dict_metanetx_id, outfile, indent=4)
    
         



def main():

    st.subheader('Primary reactant & Primary product (Use either KEGG ids or MetaNetX ids)')
    
    reactant = st.text_input(
            'reactant', value='MNXM1137670')
    product = st.text_input(
            'product', value='MNXM26')
    
    
    if st.checkbox('If metabolite not in KEGG or MetaNetX'):
            # st.subheader('test')
        add_info = st.text_area('Additional information (id: SMILES):',
                                '{"N00001":"CC(=O)O"}')
        add_info = json.loads(add_info)
    else:
        add_info = {}
    
    if st.button("Search"):
        # if session_state.button_search:
        st.subheader('Calculating optimal stoichiometry')
        st.write(reactant+" => "+ product)
        optimal_stoic(reactant,product,add_info)

if __name__ == '__main__':
    main()
        