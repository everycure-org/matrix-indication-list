"""
This is a boilerplate pipeline 'drug_lookup'
generated using Kedro 0.19.9
"""
import pandas as pd
from tqdm import tqdm
import rdflib
import networkx as nx
import matplotlib.pyplot as plt

def process_mondo_edges_nodes(mondo_nodes: pd.DataFrame, mondo_edges: pd.DataFrame)-> tuple:
    disease_nodes = mondo_nodes[mondo_nodes.category=='biolink:Disease']
    disease_edges = mondo_edges[mondo_edges['subject'].str.contains("MONDO")]
    disease_edges = disease_edges[disease_edges['object'].str.contains("MONDO")]
    disease_edges = disease_edges[disease_edges['predicate']=="biolink:subclass_of"]
    return disease_nodes, disease_edges

def build_graph(nodes, edges) -> nx.DiGraph:
    G = nx.DiGraph()
    for idx, row in tqdm(edges.iterrows(), desc='building mondo graph...'):
        G.add_edge(row['object'], row['subject'])
    return G

def get_edges(graph, name):
    return(graph.edges([name]))

def get_subclasses(G, disease_concept, output_list):
    edges = get_edges(G, disease_concept)
    if len(edges)>0:
        for e in edges:
            output_list.append(e)
            get_subclasses(G,e,output_list)

def get_all_disease_subclasses(G, top_level_disease_concept):
    output_list = []
    edges = get_edges(G, top_level_disease_concept)
    if len(edges)>0:
        for e in edges:
            output_list.append(e)
            get_subclasses(G, e, output_list)
    return output_list
    
def lookup_drugs_that_treat_disease_and_subclasses(name: str, ground_truth_list: pd.DataFrame, mondo_edges: pd.DataFrame, mondo_nodes: pd.DataFrame) -> list[str]:
    """
    Args:
        - name(str): the name of the disease being looked up
        - groundTruthList(pd.DataFrame): the ground truth list with 
          indications and contraindictions. needs to have columns:
          "disease ID", "drug ID", and "indication"

    Returns:
        - drug_IDs (list[str]): a list of drugs that treat the
          named disease and all subclasses of the named disease.
    """
    drug_IDs = []
    print(f"looking up drugs treating {name} and its subclasses...")
    disease_nodes, disease_edges = process_mondo_edges_nodes(mondo_nodes, mondo_edges)
    G = build_graph(disease_nodes, disease_edges)
    head = G.nodes['MONDO:0000001']
    #list_drug_nodes = list(i for i in ground_truth_list['drug ID'])
    list_disease_nodes = list(i for i in ground_truth_list['disease ID'])
    
    if name in list_disease_nodes and name in G:
        print(f"found {name} in indication list.")
        store_dict = {}
        head = G.nodes['name']
        print(f"generating list of disease subclasses to {name}")
        output_list = get_all_disease_subclasses(G, head)
        print(output_list)

    return drug_IDs



# Recursive downfill
def downfill(graph: nx.DiGraph, drug: str, disease: str, curr: pd.DataFrame, mondoNodes, inheritanceString: str):
    #print("downfilling treats edge between ", drug, " and ", disease)
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
        
    children = get_edges(graph, disease)
    if len(children)==0:
        return
    else:
        child_diseases = [x[1] for x in list(children)]
        for d in child_diseases:
            downfill(graph, drug, d, curr, mondoNodes, inheritanceString + "-->" + mondoNodes[mondoNodes.id==disease]['name'].to_string(index=False))

    
