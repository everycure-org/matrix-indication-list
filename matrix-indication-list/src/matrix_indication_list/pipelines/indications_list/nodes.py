"""
This is a boilerplate pipeline 'indications_list'
generated using Kedro 0.19.10
"""
import pandas as pd
from tqdm import tqdm
from io import StringIO
import requests
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import vertexai.preview.generative_models as generative_models
import os
import xml.etree.ElementTree as ET
import json
import zipfile
import os
import string
import re
from functools import cache
from openai import OpenAI
import networkx as nx

from ontobio import OntologyFactory
from ontobio.ontol_factory import OntologyFactory

import pronto

# GLOBALS
safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
    ]

generation_config = {
    "max_output_tokens":8192,
    "temperature":0,
    "top_p":0.95
}

# TESTING PARAMS
testing = False

# ID Limit, if testing
limit = 100

# SOURCE BEGINS HERE


#################################
### STANDARDIZATION #############
#################################

def standardize_ema_rows(inList:pd.DataFrame, column_names:dict) -> pd.DataFrame:
    print(inList)
    indices_to_drop = []
    for idx, row in inList.iterrows():
        if (row['Category'] != "Human") or (row['Authorisation status'] != "Authorised"):
            indices_to_drop.append(idx)
    inList.drop(indices_to_drop, inplace=True)
    print(inList)
    inList.rename(columns={
        "International non-proprietary name (INN) / common name":column_names.get("drug_name_column"),
        "Condition / indication": column_names.get("indications_text_column")
    }, inplace=True)
    return pd.DataFrame({
        column_names.get("drug_name_column"):inList[column_names.get("drug_name_column")],
        column_names.get("indications_text_column"):inList[column_names.get("indications_text_column")],
    })

def standardize_pmda_rows(inList: pd.DataFrame, column_names:dict) -> pd.DataFrame:
    print(inList)
    indices_to_drop = []
    for idx, row in inList.iterrows():
        if (type(row['Indications']) == float) or (type(row['Active Ingredient (underlined: new active ingredient)']) == float):
            indices_to_drop.append(idx)
    inList.drop(indices_to_drop, inplace=True)
    return pd.DataFrame({
        column_names.get("drug_name_column"):inList["Active Ingredient (underlined: new active ingredient)"],
        column_names.get("indications_text_column"):inList["Indications"],
    })


@cache
def generate(input_text):
        vertexai.init(project="mtrx-wg2-modeling-dev-9yj", location="us-east1")
        model = GenerativeModel(
            "gemini-1.5-flash-001",
        )
        responses = model.generate_content(
        [input_text],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
        )
        resText = ""
        for response in responses:
            resText+=response.text
            
        return resText

def clean_empty_rows(inList, column_name) -> pd.DataFrame:
    indices_to_drop = []
    for idx, row in inList.iterrows():
        if type(row[column_name])==float:
            indices_to_drop.append(idx)
    
    inList.drop(indices_to_drop, inplace=True)
    return inList

def extract_named_diseases(inList:pd.DataFrame, column_names: dict, structured_list_prompt: str) -> pd.DataFrame:
    if testing:
        print("TESTING!")

    inList = clean_empty_rows(inList, column_names.get("indications_text_column"))
    # New Fields to Add
    diseases_treated = []
    print(column_names)
    # Fetch Columns
    indications_data = list(inList[column_names.get("indications_text_column")])
    active_ingredients_data = list(inList[column_names.get("drug_name_column")])
    for index, item in tqdm(enumerate(indications_data), total=(limit if testing else len(indications_data))):
        if  (index < limit) or not testing:
            try:
                prompt = structured_list_prompt + item
                diseases_treated.append(generate(prompt))
            except Exception as e:
                print(e)
                diseases_treated.append("LLM EXTRACTION ERROR")

    if testing:
        inList = inList.head(limit)
    inList[column_names.get("disease_list_column")]=diseases_treated


    return inList

def flatten_list (inList: pd.DataFrame, column_names:dict) -> pd.DataFrame:
    drug_names_new = []
    indications_text_new = []
    structured_lists_new = []
    disease_names_new = []

    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="flattening list"):
        disease_list_curr = row[column_names.get("disease_list_column")]
        if type(disease_list_curr) != float and disease_list_curr!="LLM EXTRACTION ERROR" and disease_list_curr.strip()!="None":
            indications_list = disease_list_curr.split("|")
            for item in indications_list:
                drug_names_new.append(row[column_names.get("drug_name_column")])
                indications_text_new.append(row[column_names.get("indications_text_column")])
                structured_lists_new.append(row[column_names.get("disease_list_column")])
                disease_names_new.append(item)

    data = {
        column_names.get("drug_name_column") : drug_names_new,
        column_names.get("indications_text_column") : indications_text_new,
        column_names.get("disease_name_column") : disease_names_new,
    }
    
    return pd.DataFrame(data)

def clean_list (inList: pd.DataFrame, column_names: dict, problem_strings:list[str], regex_sub_pattern: str):
    indices_to_drop = []
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="removing problematic entries..."):
        diseaseCol = column_names.get("disease_name_column")
        disease_name = row[diseaseCol]
        if disease_name in problem_strings or type(disease_name) == float:
            indices_to_drop.append(idx)
        else:
            disease_name = re.sub(regex_sub_pattern, '', disease_name.lower())
            inList.loc[idx, diseaseCol] = disease_name
    inList.drop(indices_to_drop, inplace=True)
    return inList

@cache
def nameres(itemRequest:str) -> str:
    returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
    resolvedCurie = returned.curie
    resolvedLabel = returned.label
    return resolvedCurie, resolvedLabel

def get_curie(string, biolink_type, limit, autocomplete:str):
    itemRequest = f"https://name-resolution-sri.renci.org/lookup?string={string}&autocomplete={autocomplete}&offset=0&limit={limit}&biolink_type={biolink_type}"
    return nameres(itemRequest)

def clean_bad_entries (inList: pd.DataFrame, column_name: str, error_string: str):
    indices_to_drop = []
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="cleaning bad entries..."):
        if row[column_name] == error_string:
            indices_to_drop.append(idx)
    
    inList.drop(indices_to_drop, inplace=True)

    return inList

def resolve_concepts (inList: pd.DataFrame, column_to_resolve: str, out_column_ids: str, out_column_labels: str, biolink_type:str) -> pd.DataFrame:
    ids = []
    labels = []
    err_msg = "ERROR"
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="resolving entities from list..."):
        concept_str = row[column_to_resolve]
        if not testing or idx < limit:
            success = False
            attempts = 0
            while not success:
                if attempts >= 5:
                    ids.append(err_msg)
                    labels.append(err_msg)
                    print(f"too many failed requests for item {concept_str}")
                    success = True
                try:
                    curie, label = get_curie(string=concept_str, biolink_type=biolink_type, limit=30, autocomplete="false")
                    ids.append(curie[0])
                    labels.append(label[0])
                    success = True
                except Exception as e:
                    attempts += 1
    if testing:
        inList = inList.head(limit) 
    inList[out_column_ids] = ids
    inList[out_column_labels] = labels
    return clean_bad_entries(inList, out_column_ids, err_msg)

def check_nameres_single_entry(input_disease: str, id_label: str, params: dict) -> str:
    """
    Args: 
        inputDisease (str): the name of the disease extracted from indications text using LLMs
        params (dict): LLM parameters

    Returns:
        str: the ID of the disease as interpreted by the LLM, or "NONE"

    """

    prompt = f"{params.get('prompt')} Concept 1: {input_disease}. Concept 2: {id_label}"
    print(prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    output = client.chat.completions.create(
            model=params.get('model'),
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_disease}
                ],
            temperature= params.get('temperature')
        )
    response = output.choices[0].message.content
    print(response)
    return response

def check_nameres_llm(inList: pd.DataFrame, concept_name_column: str, nameres_label_column: str, params:dict, llm_opinion_column: str):
    tags = []
    cache = {}
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="applying LLM ID check"):
        concept = row[concept_name_column]
        nameres_label= row[nameres_label_column]
        if concept in cache:
            tags.append(cache[concept])
        else:
            try:
                llm_id = check_nameres_single_entry(concept, nameres_label, params.get("model_params"))
                cache[concept] = llm_id
                tags.append(llm_id)
            except:
                tags.append("ERROR")
        
    inList[llm_opinion_column]=tags
    return inList

def choose_best_id (concept: str, ids: list[str], labels: list[str], params: dict) -> str:
    ids_and_names = []
    for idx, item in enumerate(ids):
        ids_and_names.append(f"{idx+1}: {item} ({labels[idx]})")   
    ids_and_names = ";\n".join(ids_and_names)
    prompt = f"{params.get('prompt')} "
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    output = client.chat.completions.create(
            model=params.get('model'),
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content":  f"Disease Concept: {concept}. \r\n\n Options: {ids_and_names}"}
                ],
            temperature= params.get('temperature')
        )
    return output.choices[0].message.content

def llm_improve_ids(inList: pd.DataFrame, concept_column_name: str, params: dict, biolink_type: str, first_attempt_column_name: str, llm_decision_column: str, new_best_id_column: str ):
    
    print("Improving IDs with LLM best-choice selection")
    new_ids = []
    cache = {}
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="Using LLM to choose best of top 30 nameres hits for each flagged entry"):
        concept = row[concept_column_name]
        if row[llm_decision_column] == "TRUE":
            new_ids.append(row[first_attempt_column_name])
        else:
            if concept in cache:
                new_ids.append(cache[concept])
            else:
                try:
                    ids, labels = get_curie(string=concept, biolink_type=biolink_type, limit=30, autocomplete="false")
                    best_id = choose_best_id(concept, ids, labels, params.get('model_params'))
                    # append and cache best id from LLM
                    new_ids.append(best_id)
                    cache[concept]=best_id
                except Exception as e:
                    print(e)
                    new_ids.append("ERROR")  
    inList[new_best_id_column] = new_ids
    return clean_bad_entries(clean_bad_entries(inList, new_best_id_column, "ERROR"), new_best_id_column, "NONE")

@cache
def normalize(item: str):
    """
    Args:
        item (str): ontological ID of item
    Returns:
        tuple (str): normalized ID and label
    """
    item_request = f"https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes?curie={item}&conflate=true&drug_chemical_conflate=true&description=false&individual_types=false"    
    success = False
    failedCounts = 0
    while not success:
        try:
            response = requests.get(item_request)
            output = json.loads(response.text)
            id = output[item]['id']
            returned_id = id['identifier']
            returned_label = id['label']
            
            success = True
        except Exception as e:
            failedCounts += 1
        if failedCounts >= 5:
            return "Error", "Error"
    return returned_id, returned_label

def add_normalized_llm_tag_ids(inList: pd.DataFrame, in_column: str, out_column_id: str, out_column_label: str) -> pd.DataFrame:
    ids_out=[]
    labels_out=[]
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="normalizing tags"):
        id_in = row[in_column]
        #print(disease_id)
        if id_in == "NONE":
            ids_out.append("NONE")
            labels_out.append("NONE")
        else:
            try:
                id, label = normalize(id_in)
                ids_out.append(id)
                labels_out.append(label)
            except Exception as e:
                #print(e)
                ids_out.append("ERROR")
                labels_out.append("ERROR")
    try:
        inList[out_column_id] = ids_out
        inList[out_column_label] = labels_out
    except Exception as e:
        print(e)
    
    return clean_bad_entries(inList, out_column_id, "ERROR")


def deduplicate_entities(inList: pd.DataFrame, drug_id_column: str, disease_id_column: str, deduplication_column: str) -> pd.DataFrame:
    drug_to_disease = []
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="creating merge string for each entry"):
        drug_to_disease.append("|".join([row[drug_id_column], row[disease_id_column]]))
    inList[deduplication_column]=drug_to_disease
    inList.drop_duplicates(subset=deduplication_column, keep='first')
    return inList



def join_lists(fda_list: pd.DataFrame, ema_list: pd.DataFrame, pmda_list: pd.DataFrame, column_names:dict) -> pd.DataFrame:
    fda_list_to_merge = pd.DataFrame(data={ 
        column_names.get("llm_normalized_id_column_drug"):fda_list[column_names.get("llm_normalized_id_column_drug")],
        column_names.get("llm_normalized_label_column_drug"):fda_list[column_names.get("llm_normalized_label_column_drug")],
        column_names.get("llm_normalized_id_column_disease"):fda_list[column_names.get("llm_normalized_id_column_disease")],
        column_names.get("llm_normalized_label_column_disease"):fda_list[column_names.get("llm_normalized_label_column_disease")],
        column_names.get("deduplication_column"):fda_list[column_names.get("deduplication_column")],
    })
    ema_list_to_merge = pd.DataFrame(data={ 
        column_names.get("llm_normalized_id_column_drug"):ema_list[column_names.get("llm_normalized_id_column_drug")],
        column_names.get("llm_normalized_label_column_drug"):ema_list[column_names.get("llm_normalized_label_column_drug")],
        column_names.get("llm_normalized_id_column_disease"):ema_list[column_names.get("llm_normalized_id_column_disease")],
        column_names.get("llm_normalized_label_column_disease"):ema_list[column_names.get("llm_normalized_label_column_disease")],
        column_names.get("deduplication_column"):ema_list[column_names.get("deduplication_column")],
    })
    pmda_list_to_merge = pd.DataFrame(data={ 
        column_names.get("llm_normalized_id_column_drug"):pmda_list[column_names.get("llm_normalized_id_column_drug")],
        column_names.get("llm_normalized_label_column_drug"):pmda_list[column_names.get("llm_normalized_label_column_drug")],
        column_names.get("llm_normalized_id_column_disease"):pmda_list[column_names.get("llm_normalized_id_column_disease")],
        column_names.get("llm_normalized_label_column_disease"):pmda_list[column_names.get("llm_normalized_label_column_disease")],
        column_names.get("deduplication_column"):pmda_list[column_names.get("deduplication_column")],
    })
    merged_list = pd.concat([fda_list_to_merge, ema_list_to_merge, pmda_list_to_merge])
    clean_df = merged_list.drop_duplicates(subset=[column_names.get("deduplication_column")])
    return clean_df


## MONDO DOWNFILL

def get_edges(graph, name):
    return(graph.edges([name]))

# Recursive downfill
def downfill(graph: nx.DiGraph, drug_id: str, drug_label:str, disease_id: str, disease_label: str, curr: pd.DataFrame, mondoNodes, cache: dict, column_names: dict):
    """
    Args:
        graph (nx graph): mondo ontology digraph
        drug_id (str): id of drug that treats disease. Comes from top level row.
        drug_label(str): label for drug that treats disease. Comes from top level row.
        disease_id (str): disease id that is child of an upstream disease. Recursively generated.
        disease_label (str): label associated with disease id. Comes from calling mondoNodes[mondoNodes.id==disease_id]['name'].to_string(index=False)
        curr (pd.DataFrame): current version of dataframe mid-recursion.
        mondoNodes (pd.DataFrame): list of nodes in Mondo used to get labels.
        cache (dict): stores results so that there is no re-running.
        column_names (dict): names of columns in dataframe.
    
    """

    try:
        new_row = pd.Series({
            column_names.get("llm_normalized_id_column_drug"):drug_id,
            column_names.get("llm_normalized_label_column_drug"):drug_label,
            column_names.get("llm_normalized_id_column_disease"):disease_id,
            column_names.get("llm_normalized_label_column_disease"):disease_label,
            column_names.get("deduplication_column"): "|".join([drug_id, disease_id]),
            column_names.get("downfilled_true_false_column"):True
        })
        curr.loc[len(curr)] = new_row
        #curr = pd.concat([curr, pd.DataFrame([new_row])], ignore_index=True)
        #print(curr.tail)
        # curr.loc[len(curr)] = [drug + "|" + disease, 
        #                             mondoNodes[mondoNodes.id==disease]['name'].to_string(index=False), 
        #                             drugInfo['drug ID Label'],
        #                             drug, 
        #                             disease, 
        #                             drugInfo['active ingredients in therapy'], 
        #                             inheritanceString,
        #                             True] 
    except Exception as e:
        print(e)

    if disease_id in cache:
        children = cache[disease_id]
    else:
        children = get_edges(graph, disease_id)
        cache[disease_id] = children
    if len(children)==0:
        return
    elif len(children)<100:
        child_diseases = [x[1] for x in list(children)]
        for d in child_diseases:
            downfill(graph, drug_id, drug_label, d, mondoNodes[mondoNodes.id==d]['name'].to_string(index=False), curr, mondoNodes, cache, column_names)

def downfill_list_mondo(inList: pd.DataFrame, mondo_edges: pd.DataFrame, mondo_nodes: pd.DataFrame, column_names: dict) -> pd.DataFrame:
    K = (False for idx, row in inList.iterrows())
    inList[column_names.get("downfilled_true_false_column")] = list(K)
    # build graph from mondo
    print("importing mondo content and filtering nodes...")
    disease_nodes = mondo_nodes[mondo_nodes.category=='biolink:Disease']
    disease_edges = mondo_edges[mondo_edges['subject'].str.contains("MONDO")]
    disease_edges = disease_edges[disease_edges['object'].str.contains("MONDO")]
    disease_edges = disease_edges[disease_edges['predicate']=="biolink:subclass_of"]
    G = nx.DiGraph()
    print("building graph...")
    for idx, row in tqdm(disease_edges.iterrows()):
        G.add_edge(row['object'], row['subject'])
    cache = {}
    print("Downfilling based on MONDO hierarchy...")
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="downfilling with MONDO..."):
        disease = row[column_names.get("llm_normalized_id_column_disease")]
        if disease in cache:
            children = cache[disease]
        else:
            children = get_edges(G, disease)
            cache[disease] = children
        if len(children)>0 and len(children) < 100:
            child_diseases = [x[1] for x in list(children)]
            #print(child_diseases)
            for new_disease in child_diseases:
                downfill(G, row[column_names.get("llm_normalized_id_column_drug")], row[column_names.get("llm_normalized_label_column_drug")], new_disease, disease_nodes[disease_nodes.id==new_disease]['name'].to_string(index=False), inList, disease_nodes, cache, column_names)

    clean_df = inList.drop_duplicates(subset=[column_names.get("deduplication_column")])

    return clean_df




# def downfill_list_mondo(inList:pd.DataFrame, column_names:dict) -> pd.DataFrame:
#     #print("Building mondo ontology...")
#     #mondo = pronto.Ontology('http://purl.obolibrary.org/obo/mondo.owl')
#     print("Built mondo ontology.")
#     for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="downfilling from MONDO entries..."):
#         disease_id = row[column_names.get("llm_normalized_id_column_disease")]
#         if "MONDO" in disease_id:
#             children = get_mondo_children_pronto(disease_id, mondo)
#             for child in children:
#                 print(child.id)
#                 new_row = {
#                     column_names.get("llm_normalized_id_column_drug"):row[column_names.get("llm_normalized_id_column_drug")],
#                     column_names.get("llm_normalized_label_column_drug"):row[column_names.get("llm_normalized_label_column_drug")],
#                     column_names.get("llm_normalized_id_column_disease"):child.id,
#                     column_names.get("llm_normalized_label_column_disease"):row[column_names.get("llm_normalized_label_column_disease")],
#                     column_names.get("deduplication_column"):row[column_names.get("deduplication_column")],
#                     }
#                 inList.append(new_row, ignore_index=True)

        