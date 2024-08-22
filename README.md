# novoStoic2.0
novoStoic2.0: Integrated Pathway Design Tool with Thermodynamic Considerations and Enzyme Selection

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

## Remaining data can be taken from the scholarsphere psu link [here](https://pennstateoffice365-my.sharepoint.com/personal/vuu10_psu_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fvuu10%5Fpsu%5Fedu%2FDocuments%2Fphd%2F2024%2FnovoStoic2%5Fmetanetx%5Ffinal](https://scholarsphere.psu.edu/resources/a82e671a-cf7f-4609-afdd-a5ade27a4fab)

## creating conda environment
- create a conda environment using: `conda create --prefix pathwaydesign`
- activate the created environment using: `conda activate pathwaydesign`
- install rdkit using: `pip install rdkit` 
- install streamlit using: `pip install streamlit`

## Steps to run streamlit interface locally

run the following on terminal after activating the conda environment `streamlit run Home.py`
