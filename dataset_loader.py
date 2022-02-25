import pandas as pd
from torch_geometric.transforms import LocalDegreeProfile

def load_dataframes():

    base_path = 'datasets/'

    ddi_path = f'{base_path}DRUG_INTERACTION_graphFile_withClass.csv'
    dc_path = f'{base_path}TwoComb.csv'
    all_path = f'{base_path}AllComb.csv'

    dt_path = f'{base_path}node_features/Drugs_Targets_Onehot.csv'
    w2v_path = f'{base_path}node_features/word2vec.csv'
    sev_path = f'{base_path}node_features/sideEffectVec.csv'
    iv_path = f'{base_path}node_features/indicationsVec.csv'
    fin_path = f'{base_path}node_features/Drug_finger.csv'
    nv_path = f'{base_path}node_features/Node2Vec_DCC.csv'


    dt_df = pd.read_csv(dt_path)
    w2v_df = pd.read_csv(w2v_path)
    all_df = pd.read_csv(all_path)
    ddi_df = pd.read_csv(ddi_path)
    se_df = pd.read_csv(sev_path)
    in_df = pd.read_csv(iv_path)
    fin_df = pd.read_csv(fin_path)
    nv_df = pd.read_csv(nv_path)
    all_df.columns = ['d1', 'd2']

    # loading unique nodes list and dict
    drug_list = list()
    drug_dict = dict()

    count = 0
    for edge in all_df.values:
        n1 = edge[0]
        n2 = edge[1]
        if n1 not in drug_list:
            drug_list.append(n1)
            drug_dict[n1] = count
            count+=1
        if n2 not in drug_list:
            drug_list.append(n2)
            drug_dict[n2] = count
            count+=1

    # loading nodes with w2v features
    temp_drugs = []
    temp_vector = []
    temp_drugs_no = []
    for each in w2v_df.values:
        drug = each[0]
        str_vector = [num for num in str(each[1]).replace('[', '').replace(']', '').replace('\n', '').split(' ')]
        vector = []
        for num in str_vector:
            if num != '':
                vector.append(float(num))
        temp_drugs.append(drug)
        temp_vector.append(vector)
        temp_drugs_no.append(drug_dict[drug])

    t = pd.DataFrame(temp_vector)
    t.columns = [f'c{i}' for i in range(1, len(temp_vector[0])+1)]
    t['drugs'] = pd.DataFrame(temp_drugs)
    t['drugsNo'] = pd.DataFrame(temp_drugs_no)
    t = t.sort_values('drugsNo')
    t.index = t.drugsNo
    final_w2v_df = t.copy(deep=True)

    # loading nodes with finger features
    temp_drugs = []
    temp_vector = []
    temp_drugs_no = []

    for each in fin_df.values:
        drug = each[1]
        vector = [int(num) for num in str(each[2]).replace('[', '').replace(']', '').split(', ')]
        temp_drugs.append(drug)
        temp_vector.append(vector)
        temp_drugs_no.append(drug_dict[drug])

    t = pd.DataFrame(temp_vector)
    t.columns = [f'c{i}' for i in range(1, len(temp_vector[0])+1)]
    t['drugs'] = pd.DataFrame(temp_drugs)
    t['drugsNo'] = pd.DataFrame(temp_drugs_no)
    t = t.sort_values('drugsNo')
    t.index = t.drugsNo
    final_fin_df = t.copy(deep=True)

    # loading nodes with onehot target features
    dt_df

    temp_drugs = []
    temp_vector = []
    temp_drugs_no = []

    t = dt_df.copy(deep=True)

    for each in t.values:
        drug = each[0]
        if drug not in drug_list:
            continue
        temp_drugs_no.append(drug_dict[drug])
        temp_drugs.append(drug)


    t = t.drop(columns='DCC_ID')
    t['drugs'] = pd.DataFrame(temp_drugs)
    t['drugsNo'] = pd.DataFrame(temp_drugs_no)
    t = t.sort_values('drugsNo')
    t.index = t.drugsNo

    final_dt_df = t.copy(deep=True)

    # loading node with sideEffect features
    temp_drugs = []
    temp_vector = []
    temp_drugs_no = []

    t = se_df.copy(deep=True)

    for each in t.values:
        drug = each[0]
        if drug not in drug_list:
            continue
        temp_drugs_no.append(drug_dict[drug])
        temp_drugs.append(drug)

    t = t.drop(columns='DCC_ID')
    t['drugs'] = pd.DataFrame(temp_drugs)
    t['drugsNo'] = pd.DataFrame(temp_drugs_no)
    t = t.sort_values('drugsNo')
    t.index = t.drugsNo

    final_se_df = t.copy(deep=True)

    # loading nodes with indications features
    temp_drugs = []
    temp_vector = []
    temp_drugs_no = []

    t = in_df.copy(deep=True)

    for each in t.values:
        drug = each[0]
        if drug not in drug_list:
            continue
        temp_drugs_no.append(drug_dict[drug])
        temp_drugs.append(drug)

    t = t.drop(columns='DCC_ID')
    t['drugs'] = pd.DataFrame(temp_drugs)
    t['drugsNo'] = pd.DataFrame(temp_drugs_no)
    t = t.sort_values('drugsNo')
    t.index = t.drugsNo

    final_in_df = t.copy(deep=True)

    # loading nodes with node2vec features
    temp_drugs = []
    temp_vector = []
    temp_drugs_no = []
    for each in nv_df.values:
        drug = each[0]
        str_vector = [num for num in str(each[1]).replace('[', '').replace(']', '').replace("'", '').replace('\n', '').split(', ')]
        vector = []
        for num in str_vector:
            if num != '':
                vector.append(float(num))
        temp_drugs.append(drug)
        temp_vector.append(vector)
        temp_drugs_no.append(drug_dict[drug])

    t = pd.DataFrame(temp_vector)
    t.columns = [f'c{i}' for i in range(1, len(temp_vector[0])+1)]
    t['drugs'] = pd.DataFrame(temp_drugs)
    t['drugsNo'] = pd.DataFrame(temp_drugs_no)
    t = t.sort_values('drugsNo')
    t.index = t.drugsNo
    final_nv_df = t.copy(deep=True)

    return final_dt_df, final_w2v_df, final_nv_df, final_fin_df, final_in_df, final_se_df, all_df