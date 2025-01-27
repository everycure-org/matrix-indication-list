"""
This is a boilerplate pipeline 'fda_indications'
generated using Kedro 0.19.9
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

testing = True
limit = 1000

def generate(input_text, safety_settings, generation_config):
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

def strip_spaces(myString):
    _RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
    _RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")
    myString = _RE_COMBINE_WHITESPACE.sub(" ", myString)
    myString = _RE_STRIP_WHITESPACE.sub("", myString)
    return myString

def unzip_file(zip_path, extract_to_folder):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"The file {zip_path} does not exist.")
    os.makedirs(extract_to_folder, exist_ok=True) 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
       #print(f"Extracted all contents to {extract_to_folder}")

def extract_active_ingredient(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'fda': 'urn:hl7-org:v3'}
    active_ingredients = []
    for ingredient in root.findall(".//fda:activeMoiety/fda:name", ns):
        active_ingredients.append(ingredient.text)
    return active_ingredients

def get_spl_section(filepath: str, section_code: str) -> str | None:
    """
    Extract text from an FDA Structured Product Label (SPL) XML file based on section code.
    
    Args:
        filepath (str): Path to the SPL XML file
        section_code (str): The SPL section code to search for
        
    Returns:
        str | None: The text content of the specified section if found, None otherwise
        
    Example:
        text = get_spl_section("label.xml", "34066-1")  # Get CLINICAL STUDIES section
    """
    try:
        # Parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Define namespace mapping (SPL uses HL7 namespace)
        ns = {'hl7': 'urn:hl7-org:v3'}
        
        # Search for section with matching code
        xpath = f".//hl7:section[hl7:code[@code='{section_code}']]//hl7:text"
        section = root.find(xpath, namespaces=ns)
        
        if section is not None:
            # Extract and clean text content
            text = ''.join(section.itertext()).strip()
            return text
        
        return None
        
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")
    except Exception as e:
        raise Exception(f"Error processing SPL file: {e}")

def getIndications(xmlfilepath):
    tree = ET.parse(xmlfilepath)
    root = tree.getroot()
    ns = {'hl7': 'urn:hl7-org:v3'}
    sections = root.findall('.//hl7:section', namespaces=ns)
    for section in sections:
        codeSection = section.find('.//hl7:code', namespaces=ns)
        code = codeSection.get('code') if codeSection is not None else "no code"
        if code == "34067-9":
            text_elem = section.find('.//hl7:text', namespaces=ns)
            try:
                text_content = ''.join(text_elem.itertext()).strip()
            except:
                print('text_elem was empty')
                return ""
            return strip_spaces(text_content.strip(string.whitespace.replace(" ", "")))
        else:
            text_elem = None    
    return None

def getPediatricConsiderations(xmlfilepath):
    indicationsNameTable = ['Indications','INDICATIONS', "INDICATIONS AND USAGE", "Indications and Usage", 'INDICATIONS ', 'Indications and usage', 'INDICATIONS:', 'INDICATIONS & USAGE', 'INDICATIONS AND USAGE:', 'INDICATIONS AND USAGE ', 'INDICATIONS AND USE', '1 INDICATIONS AND USAGE']
    tree = ET.parse(xmlfilepath)
    root = tree.getroot()
    ns = {'hl7': 'urn:hl7-org:v3'}
    sections = root.findall('.//hl7:section', namespaces=ns)
    for section in sections:
        codeSection = section.find('.//hl7:code', namespaces=ns)
        code = codeSection.get('code') if codeSection is not None else "no code"
        if code == "34067-9":
            text_elem = section.find('.//hl7:text', namespaces=ns)
            text_content = ''.join(text_elem.itertext()).strip()
            return strip_spaces(text_content.strip(string.whitespace.replace(" ", "")))
        else:
            text_elem = None
        
    return None

def get_indications_codes(xmlfilepath):
    print("Finding indications for ", xmlfilepath)
    tree = ET.parse(xmlfilepath)
    root = tree.getroot()
    ns = {'hl7': 'urn:hl7-org:v3'}
    sections = root.findall('.//hl7:code', namespaces=ns)
    for code in sections:
        print(code.get('code'))

def mine_labels(dir: str) -> pd.DataFrame:
    indicationsList = []
    ingredientsList = []
    counts = 0
    foundCounts = 0
    notFoundCounts = 0
    dirs = []
    # TODO: automatically find, download, unzip all of the dailymed folders
    labelFolders = ["prescription_1/", "prescription_2/", "prescription_3/", "prescription_4/", "prescription_5/"]
    
    for label in labelFolders:
        dirs.append(dir+label)

    for directory in (dirs):
        for files in tqdm(os.listdir(directory), desc=f"reading directory {directory}"):
            if files.endswith(".zip"):
                fpath = directory + files
                fileRoot = files.replace(".zip","")
                dest = directory + fileRoot
                try:
                    unzip_file(fpath,dest)
                except:
                    #print("failed to unzip file ", fpath)
                    continue
                xmlfile=""
                for contents in os.listdir(dest):
                    if contents.endswith(".xml"):
                        xmlfile=contents.replace("._","")
                xmlfilepath = dest+"/"+xmlfile
                indications = getIndications(xmlfilepath)
                active_ingredients = extract_active_ingredient(xmlfilepath)
                for ind, item in enumerate(active_ingredients):
                    active_ingredients[ind]=item.upper()
                ingredientsList.append(set(active_ingredients))
                if indications is not None:
                    indicationsList.append(indications)
                    foundCounts += 1
                    #print(foundCounts, " indications successfully found so far")
                else:
                    notFoundCounts += 1
                    #print(notFoundCounts, " indications not found so far, failed to find for ", files)
                    indicationsList.append("")
                counts +=1
        
    print("finished ingesting indications")
    data = pd.DataFrame({'active ingredient':ingredientsList, 'indications':indicationsList})
    return data

def extract_fda_indications(inputList: pd.DataFrame, prompt: str) -> pd.DataFrame:
    #############################################
    ## GEMINI STUFF #############################
    #############################################
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
    }
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
    indicationsData = list(inputList['indications'])
    activeIngredientsData = list(inputList['active ingredient'])
    print(len(indicationsData), ' indications sections found')
    #############################################
    ## MAIN SECTION #############################
    #############################################
    diseasesTreated = []
    therapyActiveIngredients = []
    originalText = []
    cache = {}
    for index, item in tqdm(enumerate(indicationsData), total=(limit if testing else len(indicationsData))):
        if (testing and index < limit) or not testing:
            try:
                #input_text = f"Produce a list of diseases treated in the following therapeutic indications list. Please format the list as: \'item1|item2|...|itemN\'. Do not include any other text in the response. If no diseases are treated, return \'None\'. If the drug is only used for diagnostic or procedural purposes, return \'non-therapeutic\'. START TEXT HERE:"
                curr_prompt = prompt + item
                if curr_prompt in cache:
                    response = cache[curr_prompt]
                else:
                    response = generate(curr_prompt, safety_settings, generation_config)
                    cache[curr_prompt] = response
                diseasesTreated.append(response)
                therapyActiveIngredients.append(activeIngredientsData[index])
                originalText.append(item)
            except Exception as e:
                print(e)
                diseasesTreated.append("LLM ingest returned error")
                therapyActiveIngredients.append(activeIngredientsData[index])
                originalText.append(item)

    data = pd.DataFrame({'active ingredient(s)':therapyActiveIngredients,
                         'original text':originalText, 
                         'diseases treated': diseasesTreated,
                         })
    return data

@cache
def getCurie_Disease(name):
    itemRequest = 'https://name-resolution-sri.renci.org/lookup?string=' + name + '&autocomplete=false&offset=0&limit=30&biolink_type=DiseaseOrPhenotypicFeature'
    returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
    resolvedName = returned.curie
    resolvedLabel = returned.label
    return resolvedName, resolvedLabel

@cache
def getCurie_Drug(name):
    itemRequest = 'https://name-resolution-sri.renci.org/lookup?string=' + name + '&autocomplete=false&offset=0&limit=10&biolink_type=ChemicalOrDrugOrTreatment'
    returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
    resolvedName = returned.curie
    resolvedLabel = returned.label
    return resolvedName, resolvedLabel

def preferRXCUI(curieList:list[str], labelList:list[str]) -> tuple:
    """
    Args: 
        curieList (list[str]): list of Curie IDs
        labelList (list[str]): list of labels for respective Curie IDs

    Returns:
        tuple: first Curie ID that is in RXCUI and associated label, or just first curie and label if no RXCUI.

    """
    for idx, item in enumerate(curieList):
        if "RXCUI" in item:
            return item, labelList[idx]
    return curieList[0], labelList[0]

def build_string_from_list(list):
    outString = "["
    for item in list:
        outString += item + ", "
    outString = outString[:-2] + "]"
    return outString

def build_list_fda(inputList: pd.DataFrame) -> pd.DataFrame:
    diseaseData = inputList
    diseaseLabelList = []
    diseaseIDList = []
    diseaseList = []
    drugList = []
    drugIDList = []
    drugLabelList = []
    for idx, row in tqdm(diseaseData.iterrows(), total=len(diseaseData)):
        curr_row_diseasesTreated = row['diseases treated']
        if type(curr_row_diseasesTreated)!=float:
            curr_row_drugsInTherapy = row['active ingredient(s)']
            curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').split('|')
            try:
                drugCurie,drugLabel = getCurie_Drug(curr_row_drugsInTherapy)
                drugID = drugCurie[0]
                drugIDLabel = drugLabel[0]
            except:
                print("could not identify drug: ", curr_row_drugsInTherapy)
                drugID = "NameRes Failed"
                drugIDLabel = "NameRes Failed"
            for idx2,item in enumerate(curr_row_diseaseList):
                item = item.strip().upper().replace(" \n","").replace(" (PREVENTATIVE)","").replace("'","")
                if item.upper() != "NONE" and item.upper!="LLM INGEST RETURNED ERROR" and item.upper!="NON-THERAPEUTIC":
                    curr_row_diseaseList[idx2] = item
                    try:
                        diseaseCurie,diseaseLabel = getCurie_Disease(item)
                        diseaseIDList.append(diseaseCurie[0])
                        diseaseLabelList.append(diseaseLabel[0])
                        diseaseList.append(item)
                        drugList.append(curr_row_drugsInTherapy)
                        drugIDList.append(drugID)
                        drugLabelList.append(drugIDLabel)
                    except:
                        print("error during name resolving")
    sheetData = pd.DataFrame(data=[diseaseIDList, diseaseLabelList, diseaseList, drugList, drugIDList, drugLabelList]).transpose()
    sheetData.columns = ['disease IDs', 'disease ID labels', 'disease treated', 'active ingredients in therapy', 'drug ID', 'drug ID Label']
    return sheetData

def generate_tag_openai(drug_list:list, model_params:dict)-> list:
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

def check_nameres_single_entry(input_disease: str, name_res_id: str, id_label: str, params: dict) -> str:
    """
    Args: 
        inputDisease (str): the name of the disease extracted from indications text using LLMs
        params (dict): LLM parameters

    Returns:
        str: the ID of the disease as interpreted by the LLM, or "NONE"

    """

    prompt = f"{params.get('prompt')} Disease Concept 1: {input_disease}. Disease Concept 2: {id_label}"
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

def check_nameres_accuracy(inputList: pd.DataFrame, list_params: dict) -> pd.DataFrame:
    tags = []
    cache = {}
    params = list_params.get('model_params')
    name_col = list_params.get('name_column')
    for idx, row in tqdm(inputList.iterrows(), total=len(inputList), desc="applying LLM ID check"):
        disease_name = row[name_col]
        nameres_label= row['disease ID labels']
        nameres_id = row['disease IDs']
        if disease_name != "LLM INGEST RETURNED ERROR":
            if disease_name in cache:
                tags.append(cache[disease_name])
            else:
                try:
                    llm_id = check_nameres_single_entry(disease_name, nameres_id, nameres_label, params)
                    cache[disease_name] = llm_id
                    tags.append(llm_id)
                except:
                    tags.append("Error")
        else:
            tags.append("ERROR")
    
    inputList[list_params.get('output_column_name')]=tags
    return inputList

def extract_ID_llm(inputDisease: str, inputID: str, params: dict) -> str:
    """
    Args: 
        inputDisease (str): the name of the disease extracted from indications text using LLMs
        params (dict): LLM parameters

    Returns:
        str: the ID of the disease as interpreted by the LLM, or "NONE"

    """

    prompt = f"{params.get('prompt')} Disease: {inputDisease} ; ID: {inputID}"
    print(prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    output = client.chat.completions.create(
            model=params.get('model'),
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": inputDisease}
                ],
            temperature= params.get('temperature')
        )
    response = output.choices[0].message.content
    print(response)
    return response

def enrich_list_llm_ids(inputList: pd.DataFrame, list_params: dict) -> pd.DataFrame:
    llm_id_list = []
    cache = {}
    for idx, row in tqdm(inputList.iterrows(), total=len(inputList), desc="applying LLM ID tags"):
        disease_name = row[list_params.get('name_column')]
        if disease_name in cache:
            llm_id_list.append(cache[disease_name])
        else:
            try:
                llm_id = extract_ID_llm(row[list_params.get('name_column')], row['disease IDs'], list_params.get('model_params'))
                cache[disease_name] = llm_id
                llm_id_list.append(llm_id)
            except:
                llm_id_list.append(llm_id)
    
    inputList[list_params.get('output_column_name')]=llm_id_list

    return inputList

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

def add_normalized_llm_tag_ids(inList: pd.DataFrame) -> pd.DataFrame:
    ids_out=[]
    labels_out=[]
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="normalizing tags"):
        disease_id = row['llm_id']
        #print(disease_id)
        if row['llm_id'] == "NONE":
            ids_out.append("NONE")
            labels_out.append("NONE")
        else:
            try:
                id, label = normalize(disease_id)
                #print(id,label)
                ids_out.append(id)
                labels_out.append(label)
            except Exception as e:
                #print(e)
                ids_out.append("Error")
                labels_out.append("Error")
    try:
        inList['normalized ID from LLM'] = ids_out
        inList['normalized label from LLM'] = labels_out
    except Exception as e:
        print(e)

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

def add_llm_selected_best_ids(inList: pd.DataFrame, params: dict):
    print("adding LLM IDs")
    new_ids = []
    cache = {}
    id_options_cache = {}
    id_option_labels_cache={}
    options_id = []
    options_label = []
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="Using LLM to choose best of 10 nameres hits for each flagged entry"):
        disease_concept = row['disease treated']
        if row['llm_nameres_correct'] == "TRUE":
            new_ids.append(row['disease IDs'])
            options_id.append("FIRST OPTION CORRECT")
            options_label.append("FIRST OPTION CORRECT")
        else:
            if disease_concept in cache:
                new_ids.append(cache[disease_concept])
                options_id.append(id_options_cache[disease_concept])
                options_label.append(id_option_labels_cache[disease_concept])
            else:
                try:
                    ids, labels = getCurie_Disease(disease_concept)
                    best_id = choose_best_id(disease_concept, ids, labels, params.get('model_params'))
                    # append and cache best id from LLM
                    new_ids.append(best_id)
                    options_id.append("|".join(ids))
                    options_label.append("|".join(labels))
                    cache[disease_concept]=best_id
                    id_options_cache[disease_concept]="|".join(ids)
                    id_option_labels_cache[disease_concept]="|".join(labels)
                except Exception as e:
                    print(e)
                    new_ids.append("Error")
                    options_id.append("Error")
                    options_label.append("Error")
    
    inList['llm_id'] = new_ids
    inList['nameres options (ID)'] = options_id
    inList['nameres options (label)'] = options_label

    return inList