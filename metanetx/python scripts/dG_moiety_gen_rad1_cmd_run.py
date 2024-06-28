import pandas as pd
import pdb
import json
from rdkit import Chem
from tqdm import tqdm 
import numpy as np



def count_substructures(radius,molecule):
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
        env = Chem.FindAtomEnvironmentOfRadiusN(m,radius,i)
        atoms=set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())

        # only one atom is in this environment, such as O in H2O
        if len(atoms) == 0:
            atoms = {i}

        smi = Chem.MolFragmentToSmiles(m,atomsToUse=list(atoms),
                                    bondsToUse=env,canonical=True)

        if smi in smi_count:
            smi_count[smi] = smi_count[smi] + 1
        else:
            smi_count[smi] = 1
    return smi_count

def decompose_ac(db_smiles,radius=1):
    non_decomposable = []
    decompose_vector = dict()
    
    print('------ Inside the function -------')

    for cid in tqdm(db_smiles):
        # print cid
        smiles_pH7 = db_smiles[cid]
        try:
            mol = Chem.MolFromSmiles(smiles_pH7)
            mol = Chem.RemoveHs(mol)
            # Chem.RemoveStereochemistry(mol) 
            smi_count = count_substructures(radius,mol)
            decompose_vector[cid] = smi_count

        except Exception as e:
            non_decomposable.append(cid)
        
    fname = './../data_final/SMILES_moieties/decompose_vector_radius_' + str(radius) + '.json'
    
    print('------ All done writing to json ------')
    with open(fname,'w') as fp:
        json.dump(decompose_vector,fp)


        
with open('./../data_final/metab_db_final_new_no_metal.json') as mnx_met_final:
    metanetx_metab_db_final = json.load(mnx_met_final)
    
    
print('------ File reading done -------')
met_ls_final = list(metanetx_metab_db_final.keys())


metanetx_metab_db_smiles = metanetx_metab_db_final
        
print('------ Input ready to enter the function -------')
decompose_ac(metanetx_metab_db_smiles, radius = 1)