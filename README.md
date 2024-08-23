# novoStoic2.0
novoStoic2.0: Integrated Pathway Design Tool with Thermodynamic Considerations and Enzyme Selection

## Related work
1. Wang L, Upadhyay V, Maranas CD (2021) dGPredictor: Automated fragmentation method for metabolic reaction free energy prediction and de novo pathway design. PLOS Computational Biology 17(9): e1009448. https://doi.org/10.1371/journal.pcbi.1009448
2. Upadhyay, V., Boorla, V. S., & Maranas, C. D. (2023). Rank-ordering of known enzymes as starting points for re-engineering novel substrate activity using a convolutional neural network. Metabolic engineering, 78, 171–182. https://doi.org/10.1016/j.ymben.2023.06.001
3. Kumar, A., Wang, L., Ng, C.Y. et al. Pathway design using de novo steps through uncharted biochemical spaces. Nat Commun 9, 184 (2018). https://doi.org/10.1038/s41467-017-02362-x
4. Chowdhury, A., Maranas, C. Designing overall stoichiometric conversions and intervening metabolic reactions. Sci Rep 5, 16009 (2015). https://doi.org/10.1038/srep16009

### Requirements: 

1. Rdkit
2. Tensorflow 2
3. Streamlit
4. Pandas
5. Numpy
6. Keras
7. scikit-learn
8. matplotlib
9. Pulp
10. CPLEX solver
11. ChemAxon's Marvin >= 5.11
12. Openbabel

Refer the file titled _env.yaml_ for full list of depedencies

## Remaining data can be taken from the scholarsphere psu link [here](https://doi.org/10.26207/fxd2-se27)
Due to constraints of file sizes on github, we have published all the data and codes on shcholar sphere psu.

## creating conda environment
- create a conda environment using: `conda create --prefix pathwaydesign`
- activate the created environment using: `conda activate pathwaydesign`
- install rdkit using: `pip install rdkit` 
- install streamlit using: `pip install streamlit`

## Steps to run streamlit interface locally

run the following on terminal after activating the conda environment `streamlit run Home.py`
