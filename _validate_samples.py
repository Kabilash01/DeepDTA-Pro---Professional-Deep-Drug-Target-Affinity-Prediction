from rdkit import Chem

drugs = {
    'Imatinib (Gleevec)':  'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C',
    'Gefitinib (Iressa)':  'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4',
    'Erlotinib (Tarceva)': 'C#Cc1cccc(c1)Nc2ncnc3cc(c(cc23)OCCO)OCCO',
    'Sorafenib (Nexavar)': 'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1',
    'Dasatinib (Sprycel)': 'Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1',
    'Lapatinib (Tykerb)':  'CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1',
    'Sunitinib (Sutent)':  'CCN(CC)CCNC(=O)c1c(C)[nH]c(c1C)C=C1C(=O)Nc2ccc(F)cc21',
    'Nilotinib (Tasigna)': 'Cc1cn(cn1)-c1cc(NC(=O)c2ccc(C)c(Nc3nccc(n3)-c3cccnc3)c2)cc(c1)C(F)(F)F',
    'Aspirin':             'CC(=O)Oc1ccccc1C(=O)O',
    'Ibuprofen':           'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
}

for name, smi in drugs.items():
    mol = Chem.MolFromSmiles(smi)
    print(('VALID  ' if mol else 'INVALID') + ' | ' + name)
