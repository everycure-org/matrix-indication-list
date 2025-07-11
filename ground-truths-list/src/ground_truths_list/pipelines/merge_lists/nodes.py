import pandas as pd
from tqdm import tqdm
import rdflib
import networkx as nx
import matplotlib.pyplot as plt
from openai import OpenAI

def join_strings(series, delimiter=", "):
        return delimiter.join(series)


def merge_lists(inputList_fda: pd.DataFrame, inputList_ema: pd.DataFrame, inputList_pmda: pd.DataFrame) -> pd.DataFrame:
    fda_frame = inputList_fda
    ema_frame = inputList_ema
    pmda_frame = inputList_pmda

    print(len(fda_frame), " items in fda frame")
    print(len(ema_frame), " items in ema frame")
    print(len(pmda_frame), " items in pmda frame")

    df1 = pd.merge(ema_frame, pmda_frame, how="outer")
    df2 = pd.merge(df1, fda_frame, how="outer")
    df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
    #df2.to_csv("indicationList.tsv", sep="\t")
    df2['drug|disease'] = df2['disease IDs'].astype(str) + '|' + df2['drug ID']
    print (df2)
    
    
    agg_functions = {'disease ID labels': 'first', 'drug ID Label': 'first', 'drug ID': 'first', 'disease IDs': 'first', 'active ingredients in therapy':lambda x: list(x), 'disease treated': lambda x: list(x)}

    df3 = df2.groupby(df2['drug|disease']).aggregate(agg_functions)


    print(len(set(df2['disease IDs'])), " unique diseases indicated for")
    print(len(set(df2['drug ID'])), " unique drugs accounted for")
    print(len(set(df2['drug|disease'])), " unique drug->treats->disease edges")

    for i in range(len(df3)):
        if isinstance(df3.iloc[i,4][0], float) == False:
            remove_brackets = [l.replace("{","").replace("}","").replace("'",'') for l in df3.iloc[i,4]]
            unique_list = sorted(list(set(remove_brackets)))
            df3.iloc[i,4] = unique_list
        
        unique_disease_list = sorted(list(set(df3.iloc[i,5])))
        df3.iloc[i,5] = unique_disease_list

    return df2
    #df3.to_csv("indicationList.tsv", sep="\t")


## MONDO DOWNFILL

def get_edges(graph, name):
    return(graph.edges([name]))

# Recursive downfill
def downfill(graph: nx.DiGraph, drug: str, disease: str, curr: pd.DataFrame, mondoNodes, inheritanceString: str, cache: dict):
    drugInfo = curr[curr['drug ID']==drug].iloc[0]
    try:
        curr.loc[len(curr)] = [drug + "|" + disease, 
                                    mondoNodes[mondoNodes.id==disease]['name'].to_string(index=False), 
                                    drugInfo['drug ID Label'],
                                    drug, 
                                    disease, 
                                    drugInfo['active ingredients in therapy'], 
                                    inheritanceString,
                                    True] 
    except Exception as e:
        print(e)
        print(curr)
        print(curr.index)
    
    if disease in cache:
        children = cache[disease]
    else:
        children = get_edges(graph, disease)
        cache[disease] = children
    if len(children)==0:
        return
    elif len(children) < 3:
        child_diseases = [x[1] for x in list(children)]
        for d in child_diseases:
            downfill(graph, drug, d, curr, mondoNodes, inheritanceString + "-->" + mondoNodes[mondoNodes.id==disease]['name'].to_string(index=False), cache)



def mondo_downfill_operation(indicationList_merged: pd.DataFrame, mondo_edges: pd.DataFrame, mondo_nodes: pd.DataFrame) -> pd.DataFrame:
    print("importing indication list...")
    indication_list = indicationList_merged #pd.read_csv("../merge_lists/indicationList.tsv", sep='\t')

    K = (False for idx, row in indication_list.iterrows())

    indication_list['inferred_from_mondo'] = list(K)

    print("importing mondo content and filtering nodes...")

    disease_nodes = mondo_nodes[mondo_nodes.category=='biolink:Disease']
    disease_edges = mondo_edges[mondo_edges['subject'].str.contains("MONDO")]
    disease_edges = disease_edges[disease_edges['object'].str.contains("MONDO")]
    disease_edges = disease_edges[disease_edges['predicate']=="biolink:subclass_of"]


    G = nx.DiGraph()
    print("building graph...")
    for idx, row in tqdm(disease_edges.iterrows()):
        G.add_edge(row['object'], row['subject'])

    # now we have a directed graph originating at "Disease". Designate "MONDO:0000001" as HEAD.
    head = G.nodes['MONDO:0000001']
    list_drug_nodes = list(i for i in indication_list['drug ID'])
    list_disease_nodes = list(i for i in indication_list['disease IDs'])

    # NOTE: child refers to being downhill in the MONDO tree, not to human children.
    cache = {}
    print("Downfilling based on MONDO hierarchy...")
    for idx, disease in tqdm(enumerate(list_disease_nodes), total=len(list_disease_nodes)):
        if disease in cache:
            children = cache[disease]
        else:
            children = get_edges(G, disease)
            cache[disease] = children
        if len(children)>0 and len(children) < 3:
            child_diseases = [x[1] for x in list(children)]
            #print(disease, " has ", len(child_diseases) ," downhill children: ", child_diseases)
            for K in child_diseases:
                downfill(G, list_drug_nodes[idx], K, indication_list, disease_nodes, disease_nodes[disease_nodes.id==disease]['name'].to_string(index=False), cache)
    return indication_list


def generate_tag_openai(drug_list:list, model_params:dict) -> list:
    """Generates tags based on provided prompts and params through OpenAI API call.
    
    Args:
        drug_list: list- list of drugs for which tags should be generated.
        model_params: Dict - parameters dictionary for openAI API call
    Returns
        List of tags generated by the API call.
    """
    tag_list = []
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    for drug in tqdm(drug_list):
        output = client.chat.completions.create(
            model=model_params.get('model'),
            messages=[
                    {"role": "system", "content": model_params.get('prompt')},
                    {"role": "user", "content": drug}
                ],
            temperature= model_params.get('temperature')
        )
        tag_list.append(output.choices[0].message.content)
    return tag_list


