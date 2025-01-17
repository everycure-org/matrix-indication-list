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

testing = True
limit = 5000

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

# def getIndications(spl_file_path: str) -> str:
#     """
#     Extract the content of section 34067-9 from an FDA SPL XML file.
    
#     Args:
#         spl_file_path (str): Path to the SPL XML file
        
#     Returns:
#         str: The text content of section 34067-9, or None if not found
#     """
#     try:
#         # Parse the XML file
#         tree = ET.parse(spl_file_path)
#         root = tree.getroot()
        
#         # Define the namespace mapping
#         # SPL files typically use these namespaces
#         namespaces = {
#             'v3': 'urn:hl7-org:v3',
#             'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
#         }
        
#         # Find the section with code 34067-9
#         # Using xpath to find the section
#         section_xpath = ".//v3:section[.//v3:code[@code='34067-9']]"
#         section = root.find(section_xpath, namespaces)
        
#         if section is not None:
#             # Extract all text content from the section
#             text_elements = section.findall(".//v3:text", namespaces)
#             content = []
#             for text_elem in text_elements:
#                 if text_elem.text:
#                     content.append(text_elem.text.strip())
            
#             return "\n".join(content)
#         else:
#             return None
            
#     except ET.ParseError as e:
#         #print(f"Error parsing XML file: {e}")
#         return None
#     except Exception as e:
#         #print(f"An error occurred: {e}")
#         return None

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
    for index, item in tqdm(enumerate(indicationsData), total=(limit if testing else len(indicationsData))):
        if (testing and index < limit) or not testing:
            try:
                #input_text = f"Produce a list of diseases treated in the following therapeutic indications list. Please format the list as: \'item1|item2|...|itemN\'. Do not include any other text in the response. If no diseases are treated, return \'None\'. If the drug is only used for diagnostic or procedural purposes, return \'non-therapeutic\'. START TEXT HERE:"
                response = generate(prompt+item, safety_settings, generation_config)
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

def getCurie_Disease(name):
    itemRequest = 'https://name-resolution-sri.renci.org/lookup?string=' + name + '&autocomplete=false&offset=0&limit=10&biolink_type=DiseaseOrPhenotypicFeature'
    returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
    resolvedName = returned.curie
    resolvedLabel = returned.label
    return resolvedName, resolvedLabel

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

    nRows = len(list(diseaseData['diseases treated']))

    label_cache = {}
    id_cache = {}

    drug_id_cache = {}
    drug_label_cache = {}

    for index, row in tqdm(diseaseData.iterrows(), total=len(diseaseData)):
        curr_row_diseasesTreated = row['diseases treated']
        if type(curr_row_diseasesTreated)!=float:
            
            curr_row_drugsInTherapy = row['active ingredient(s)']
            curr_row_disease_ids = []
            curr_row_disease_id_labels = []
            curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').split('|')
            #print("disease list: ", curr_row_diseaseList)
            
            if curr_row_drugsInTherapy in drug_id_cache:
                print("Drug cache hit")
                drugID = drug_id_cache[curr_row_drugsInTherapy]
                drugIDLabel = drug_label_cache[curr_row_drugsInTherapy]
            else:
                try:
                    drugCurie,drugLabel = getCurie_Drug(curr_row_drugsInTherapy)
                    drugID = drugCurie[0]
                    drugIDLabel = drugLabel[0]
                    drug_id_cache[curr_row_drugsInTherapy]=drugID
                    drug_label_cache[curr_row_drugsInTherapy]=drugIDLabel
                        
                except:
                    print("could not identify drug: ", curr_row_drugsInTherapy)
                    drugID = "NameRes Failed"
                    drugIDLabel = "NameRes Failed"

            for idx2,item in enumerate(curr_row_diseaseList):
                item = item.strip().upper().replace(" \n","").replace(" (PREVENTATIVE)","").replace("'","")
                if item.upper() != "NONE" and item.upper!="LLM INGEST RETURNED ERROR" and item.upper!="NON-THERAPEUTIC":
                    if item in id_cache:
                        print("Cache Hit")
                        diseaseList.append(item)
                        diseaseIDList.append(id_cache[item])
                        diseaseLabelList.append(label_cache[item])
                        drugList.append(curr_row_drugsInTherapy)
                        drugIDList.append(drugID)
                        drugLabelList.append(drugIDLabel)
                    else:
                        curr_row_diseaseList[idx2] = item
                        #print(item)
                        try:
                            #print(item)
                            diseaseCurie,diseaseLabel = getCurie_Disease(item)
                            diseaseIDList.append(diseaseCurie[0])
                            diseaseLabelList.append(diseaseLabel[0])
                            diseaseList.append(item)
                            drugList.append(curr_row_drugsInTherapy)
                            drugIDList.append(drugID)
                            drugLabelList.append(drugIDLabel)
                            id_cache[item]=diseaseCurie[0]
                            label_cache[item]=diseaseLabel[0]

                        except:
                            print("error during name resolving")

    sheetData = pd.DataFrame(data=[diseaseIDList, diseaseLabelList, diseaseList, drugList, drugIDList, drugLabelList]).transpose()
    sheetData.columns = ['disease IDs', 'disease ID labels', 'list of diseases', 'active ingredients in therapy', 'drug ID', 'drug ID Label']
    return sheetData