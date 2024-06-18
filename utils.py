import pandas as pd 
import numpy as np 
import json 
from tqdm import tqdm

import difflib
from difflib import SequenceMatcher
import pdb

from d3flux import flux_map
import networkx as nx

import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

rules_ndp = pd.read_csv('./../../metanetx/data_final/SMILES_moieties/reaction_rules_nodup_metanetx_correction.csv', index_col=0)
rules = pd.read_csv('./../../metanetx/data_final/SMILES_moieties/reaction_rules_all_metanetx_correction.csv', index_col=0)

ms = json.load(open('./../../metanetx/data_final/SMILES_moieties/Mol_sig_MNXM_moiety_correction_dict.json'))
rule_mapping = json.load(open('./../../metanetx/data_final/SMILES_moieties/rule_mapping_metanetx.json'))

sij = json.load(open('./../../metanetx/data_final/MNXR_Sij_final.json'))
cofactors = pd.read_csv('./../../metanetx/data_final/cofactors.csv', index_col=0)

metab_db = json.load(open('./../../metanetx/data_final/metanetx_metab_db_nodup.json'))

## reading unbalanced reaction sij from the database incase an incomplete reaction from database is in pathway
nontransport_rxn_Metanetx = json.load(open('./../../metanetx/data_final/metanetx_db_sij_notransport.json'))
unbalanced_reactions_sij = {key: value for key, value in nontransport_rxn_Metanetx.items() if key not in sij}

cf = list(cofactors.index)

rules_rxn_id = rules.columns.to_list()
moieites = rules.index.to_list()

rules_unique_rxn_id = rules_ndp.columns.to_list()
moieties = rules_ndp.index.to_list()

mol_sig = pd.DataFrame(ms).fillna('0').reindex(moieites)

def parse_solution_file(file_path):
    solutions = {}
    TF = {}
    with open(file_path, 'r') as file:
        current_iteration = None
        for line in file:
            line = line.strip()
            if line.startswith('iteration'):
                current_iteration = int(line.split(',')[1])
                solutions[current_iteration] = {}
            elif line.startswith('MNXR'):
                reaction_id, flux = line.split(',')
                try:
                    solutions[current_iteration][reaction_id] = round(float(flux))
                except:
                    solutions[current_iteration][reaction_id] = round(float(flux.split(' ')[0]))
                    TF[current_iteration] = 1
    return solutions, TF

## now separating the using the molsig of subs and product with rule vector 
## to generate all possible combination of molecules and match if they are present in the database

def separate_reactants_products(reaction_dict):
    reactant_ids = []
    reactant_stoichiometry = []
    product_ids = []
    product_stoichiometry = []

    for compound, stoichiometry in reaction_dict.items():
        if compound in ['MNXM01', 'MNXM1', 'MNXM1108018']:
            continue  # Ignore specific compounds
        if stoichiometry < 0:
            reactant_ids.append(compound)
            reactant_stoichiometry.append(stoichiometry)
        elif stoichiometry > 0:
            product_ids.append(compound)
            product_stoichiometry.append(stoichiometry)

    return reactant_ids, reactant_stoichiometry, product_ids, product_stoichiometry

def remove_cofactors(rct_ids, pdt_ids):
    prim_reactant = []
    prim_pdt = []
    
    for i in rct_ids:
        if i not in cf:
            prim_reactant.append(i)
    
    
    for i in pdt_ids:
        if i not in cf:
            prim_pdt.append(i)

    return prim_reactant, prim_pdt


def are_dicts_equal(dict1, dict2):
    """
    Check if two dicts are equal irrespective of the order of keys.
    
    Args:
    - dict1: dict
    - dict2: dict
    
    Returns:
    - bool: True if dictionaries are equal, False otherwise.
    """
    # Check if the number of keys is the same
    if len(dict1) != len(dict2):
        return False
    
    # Check if all key-value pairs in dict1 are present in dict2
    for key, value in dict1.items():
        if key not in dict2 or dict2[key] != value:
            return False
    
    return True

def non_zero_elements_to_dict(series):
    """
    Convert non-zero elements in a pandas Series into a dictionary
    with index as key and value as value.
    
    Args:
    - series: pandas Series
    
    Returns:
    - non_zero_dict: dict
    """
    nzix = series[series != 0].index
    non_zero_dict = {index: value for index, value in zip(nzix, series.loc[nzix])}
    return non_zero_dict


def find_reactions_with_substrate(sij, Primary_substrate):
    rxn_with_sub = []

    for i in sij:
        temp = list(sij[i].keys())
        if Primary_substrate in temp:
            rxn_with_sub.append(i)
    
    return rxn_with_sub

def find_reactions_with_product(sij, Primary_product):
    rxn_with_pdt = []

    for i in sij:
        temp = list(sij[i].keys())
        if Primary_product in temp:
            rxn_with_pdt.append(i)
    
    return rxn_with_pdt


def test_Rxn_containing_substrate_and_product(rxn_id, pdt_rxn, sub_rxn):
    sol_with_subs = []
    sol_with_pdt = []

    for i in rxn_id:
        if i in pdt_rxn:
            sol_with_pdt.append(i)
        elif i in sub_rxn:
            sol_with_subs.append(i)

    return sol_with_subs, sol_with_pdt

def test_duplicated_rules_containing_substrate_and_product(rxn_ids, dup_rule_map, sij, Pri_subs, Pri_pdts):
    Rxn1_subs = []
    RxnL_pdt = []

    for i in rxn_ids:
        temp_l = dup_rule_map[i]
        for j in temp_l:
            temp_sij = list(sij[j].keys())
            if Pri_subs in temp_sij:
                Rxn1_subs.append([i, j])
            elif Pri_pdts in temp_sij:
                RxnL_pdt.append([i,j])

    return Rxn1_subs, RxnL_pdt


def apply_rules_on_substrate(reaction_id, flux_ls, sij, mol_sig_df, Pri_subs, Pri_pdt, Reactionrules, ms_dict):
    pdt_ms_first_step_all_permut = {}
    pdt_step1_found = {}

    for i, rids in enumerate(reaction_id):
        rxn_flux = flux_ls[i]
        rxn_sij = sij[rids]

        rxn_comp = list(rxn_sij.keys())
        rxn_stoic = list(rxn_sij.values())

        if flux_ls[i] > 0:
            rct_id, rct_s, pdt_id, pdt_s = separate_reactants_products(rxn_sij)
            pr_rct, pr_pdt = remove_cofactors(rct_id, pdt_id)
        else:
            pdt_id, pdt_s, rct_id, rct_s = separate_reactants_products(rxn_sij)
            pr_rct, pr_pdt = remove_cofactors(rct_id, pdt_id)

        reactant_new = [Pri_subs if x == pr_rct[0] else x for x in rct_id]

        reactant_side_ms = pd.Series(0, index=mol_sig_df[Pri_subs].index)
        for j, jr in enumerate(reactant_new):
            reactant_side_ms += mol_sig_df[jr].astype(int) * int(np.abs(rct_s[j]))

        product_new = list(set(pdt_id) - set(pr_pdt))

        pdt_stoic_new = pdt_s
        index_prim_pdt = pdt_id.index(pr_pdt[0])
        pdt_stoic_new.pop(index_prim_pdt)

        product_side_ms = pd.Series(0, index=mol_sig_df[Pri_pdt].index)

        for k, kp in enumerate(product_new):
            product_side_ms += mol_sig_df[kp].astype(int) * int(np.abs(pdt_stoic_new[k]))

        rxn_rule = Reactionrules[rids]

        if flux_ls[i] < 0:
            MS_pdt = reactant_side_ms - product_side_ms - rxn_rule
        else: 
            MS_pdt = reactant_side_ms - product_side_ms + rxn_rule

        pdt_ms_first_step_all_permut[rids] = MS_pdt 
        MS_pdt_dict = non_zero_elements_to_dict(MS_pdt)
        for mnxmi in ms_dict:
            if are_dicts_equal(MS_pdt_dict, ms_dict[mnxmi]):
                pdt_step1_found[rids] = mnxmi

    return pdt_ms_first_step_all_permut, pdt_step1_found


def process_pathway_solution(Step1_product_found, sij, unbalanced_metanetx_rxn_sij, Pri_substrate, flx_ls):
    pathway_solution = {}

    for ix, si in enumerate(Step1_product_found):
        sol_rxn_sij = sij[si]

        rside_id, rside_s, pside_id, pside_s = separate_reactants_products(sol_rxn_sij)
        prim_rct, prim_pdt = remove_cofactors(rside_id, pside_id)
        flx = flx_ls[ix]

        if flx > 0:
            rside_id_new = [Pri_substrate if x == prim_rct[0] else x for x in rside_id]
            pside_id_new = [Step1_product_found[si] if x == prim_pdt[0] else x for x in pside_id]

            rside_s_new = rside_s
            pside_s_new = pside_s
        else:
            pside_id_new = [Pri_substrate if x == prim_rct[0] else x for x in rside_id]
            rside_id_new = [Step1_product_found[si] if x == prim_pdt[0] else x for x in pside_id]

            rside_s_new = [x * flx for x in rside_s]
            pside_s_new = [x * flx for x in pside_s]

        sol_rxn_sij_new = {key: value for key, value in zip(rside_id_new + pside_id_new, rside_s_new + pside_s_new)}

        prim_rct_new, prim_pdt_new = remove_cofactors(rside_id_new, pside_id_new)

        last_test_f = []
        last_test_b = []
        last_test = {}

        for rx_i in unbalanced_metanetx_rxn_sij:
            temp_rside_id, temp_rside_s, temp_pside_id, temp_pside_s = separate_reactants_products(unbalanced_metanetx_rxn_sij[rx_i]['Sij'])
            temp_prim_rct, temp_prim_pdt = remove_cofactors(temp_rside_id, temp_pside_id)

            if len(temp_prim_rct) == 0 or len(temp_prim_pdt) == 0:
                pass
            else:
                if prim_pdt_new[0] in temp_prim_pdt and prim_rct_new[0] in temp_prim_pdt:
                    last_test_f.append(rx_i)
                elif prim_pdt_new[0] in temp_prim_rct and prim_rct_new[0] in temp_prim_pdt:
                    last_test_b.append(rx_i)

                last_test = {'Forward': last_test_f, 'Reverse': last_test_b}

        pathway_solution[ix] = {'Rule_id': si, 'Sij_updated': sol_rxn_sij_new, 'final_test': last_test}
    
    return pathway_solution

def process_terminal_moieties(ms_comp, moieties_ls, ms_dict):
    neg_moie = []
    pos_moie = []
    for i in ms_comp:
        if ms_comp[i] < 0:
            neg_moie.append(i)
        else:
            pos_moie.append(i)

    ms_comp_cp = {}
    pos_moie_str = []
    neg_moie_str = []
    for i in neg_moie:
        temp_moie = most_similar_string(i, pos_moie)
        if ms_comp[temp_moie] + ms_comp[i] == 0:
            pos_moie_str.append(temp_moie)
            neg_moie_str.append(i)

            ms_comp_temp = ms_comp.copy()
            ms_comp_temp.pop(temp_moie)
            ms_comp_temp.pop(i)


    pdt_found = []
    for ix, nms in enumerate(neg_moie_str):        
        temp_smiles_overlap, temp_diff1, temp_diff2 = find_string_differences(nms, pos_moie_str[ix])
        temp_pdt_found = []
        for idx in ms_comp_temp:
            if temp_diff1 in i:
                mnew = idx.replace(temp_diff1, temp_diff2)
                if mnew in moieties_ls:
                    ms_comp_updated = ms_comp_temp.copy()
                    ms_comp_updated[mnew] = ms_comp_temp[idx]
                    ms_comp_updated.pop(idx)
                    for msi in ms_dict:
                        if are_dicts_equal(ms_comp_updated, ms_dict[msi]) == True:
                            temp_pdt_found.append(msi)

        if len(temp_pdt_found) != 0:
            pdt_found.append(temp_pdt_found)
        else:
            for msi in ms_dict:
                if are_dicts_equal(ms_comp_temp, ms_dict[msi]) == True:
                    temp_pdt_found.append(msi)

        if len(temp_pdt_found) != 0:
            pdt_found.append(temp_pdt_found)
                
                                
            # pdt_found.append(temp_pdt_found)

    return pdt_found



def most_similar_string(input_string, string_list):
    max_ratio = 0
    most_similar = None

    for string in string_list:
        ratio = SequenceMatcher(None, input_string, string).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            most_similar = string

    return most_similar

def find_string_differences(string1, string2):
    """
    Find the common part and differences between two strings.

    Args:
    - string1 (str): First input string.
    - string2 (str): Second input string.

    Returns:
    - tuple: A tuple containing the common part and the differences.
    """
    # Find the differences between the two strings
    diff = difflib.ndiff(string1, string2)

    # Initialize variables to store the common and different parts
    common_part = ''
    difference1 = ''
    difference2 = ''

    # Iterate through the differences
    for item in diff:
        # If the characters are the same in both strings, add them to the common_part
        if item[0] == ' ':
            common_part += item[-1]
        # If the character is present in string1 and not in string2, add it to difference1
        elif item[0] == '-':
            difference1 += item[-1]
        # If the character is present in string2 and not in string1, add it to difference2
        elif item[0] == '+':
            difference2 += item[-1]

    return common_part, difference1, difference2

def get_substrings(string):
    parts = string.split('_')
    result = []

    # Start the loop from index 1 to exclude 'P1'
    for i in range(1, len(parts) + 1):
        result.append('_'.join(parts[:i]))

    # Remove 'P1' from the result list
    result = result[1:]

    return result
