"""
This is a boilerplate pipeline 'pmda_indications'
generated using Kedro 0.19.9
"""
import os
import google.generativeai as genai
import pandas as pd
import time
import urllib

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import vertexai.preview.generative_models as generative_models

from tqdm import tqdm

#############################################
## GEMINI STUFF #############################
#############################################

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


def extract_pmda_indications(input_list: pd.DataFrame, prompt: str) -> pd.DataFrame:
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

    #############################################
    ## END OF GEMINI STUFF ######################
    #############################################



    # ORDER OF OPERATIONS FOR THIS SCRIPT
    # 1. READ IN XLSX SHEET
    # 2. MAKE DRUG NAMES UPPER AND PUT IN LIST
    # 3. GET LIST OF INDICATIONS
    # 4a. GENERATE DISEASES FROM INDICATIONS USING LLM
    # 4b. Generate XLSX of rows consisting of drug name and list of indications interpreted from LLM.
    # 5. FOR EACH ITEM IN DRUG LIST SET:
    # 5.1 GET ALL INDICATIONS ASSOCIATED WITH ITEM, COLLECT.
    # 5.2 APPEND [DrugName, Collection] to new dataframe
    # 6. NAME RESOLVE TO GET ONTOLOGICAL MAPPING

    pmda = input_list #pd.read_csv("pmda_approvals.csv")
    drugnames = list(pmda['Active Ingredient (underlined: new active ingredient)'])
    for idx, name in enumerate(drugnames):
        try:
            drugnames[idx] = name.upper()
        except:
            drugnames[idx] = drugnames[idx]

    print(len(drugnames), 'rows of drugs')
    print(len(set(drugnames)), 'unique drugs')

    indications = pmda['Indications']

    print(len(indications), ' indications for human use')

    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import math
    ########################################
    ## WARNING: THIS COSTS MONEY TO RUN ####
    ##   THIS PORTION CONFIRMS YOUR RUN ####
    ########################################
    # print("WARNING: This section of the script uses Gemini resources - confirm that you want to run it?")
    # print("Waiting 5 seconds so you don't just hit continue without thinking...")
    # time.sleep(5)

    # successfulInput = False
    # while not successfulInput:
    #     scriptRun = input('Run the script? (yes / no)')
    #     if scriptRun == "yes":
    #         successfulInput = True
    #     elif scriptRun == "no":
    #         assert(0)
    #     else:
    #         print("Bad input. Please input ''yes'' / ''no''")
    #         successfulInput = False


    ############################################
    ## THIS PART ACTUALLY EXECUTES THE INGEST ##
    ############################################

    indicationDiseaseList = []
    drugsList = []
    originalText = []
    testing = False
    limit = 100
    for index, ind in tqdm(enumerate(indications), total=len(indications)):
        if not testing or index < limit:
            #print(index)
            if not ind or type(ind)==float:
                print("found nan value")
                indicationDiseaseList.append("NA")
                drugsList.append(drugnames[index])
                originalText.append("NA")
            else:
                drugsList.append(drugnames[index])
                originalText.append(ind)
                try:
                    response = generate(prompt+ind, safety_settings, generation_config)
                    print(response)
                    indicationDiseaseList.append(response.upper())
                except:
                    print("LLM extraction failure")
                    indicationDiseaseList.append("LLM failed to extract indications")

    sheetData = pd.DataFrame(data=[drugsList,indicationDiseaseList, originalText]).transpose()
    sheetData.columns = ['drug active ingredients', 'diseases', 'original text']
    return sheetData


import pandas as pd
import requests
from io import StringIO

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

def build_string_from_list(list):
    outString = "["
    for item in list:
        outString += item + ", "
    outString = outString[:-2] + "]"
    return outString


def build_list_pmda(inputList: pd.DataFrame) -> pd.DataFrame:
    diseaseData = inputList#pd.read_excel('../drug-disease-pairs-pmda.xlsx')

    diseaseLabelList = []
    diseaseIDList = []
    diseaseList = []
    drugList = []
    drugIDList = []
    drugLabelList = []


    for index, row in tqdm(diseaseData.iterrows(), total=len(diseaseData)):
        curr_row_diseasesTreated = row['diseases']
        if type(curr_row_diseasesTreated)!=float:
            #print(index)
            curr_row_drugsInTherapy = row['drug active ingredients']
            curr_row_disease_ids = []
            curr_row_disease_id_labels = []
            curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').replace("'","").split('|')
            #print("disease list: ", curr_row_diseaseList)
            
            try:
                drugCurie,drugLabel = getCurie_Drug(curr_row_drugsInTherapy)
                drugID = drugCurie[0]
                drugIDLabel = drugLabel[0]
                    
            except:
                print("could not identify drug: ", curr_row_drugsInTherapy)
                drugID = "NameRes Failed"
                drugIDLabel = "NameRes Failed"

            for idx2,item in enumerate(curr_row_diseaseList):
                item = item.strip().replace(" \n","").replace(" (PREVENTATIVE)","")
                curr_row_diseaseList[idx2] = item
                #print(item)
                
                try:
                    #print("item being evaluated: ",item)
                    diseaseCurie,diseaseLabel = getCurie_Disease(item)
                    #print(diseaseCurie[0])
                    diseaseIDList.append(diseaseCurie[0])
                    diseaseLabelList.append(diseaseLabel[0])
                    diseaseList.append(item)
                    drugList.append(curr_row_drugsInTherapy)
                    drugIDList.append(drugID)
                    drugLabelList.append(drugIDLabel)
                    
                except:
                    print("error during name resolving")

    sheetData = pd.DataFrame(data=[diseaseIDList, diseaseLabelList, diseaseList, drugList, drugIDList, drugLabelList]).transpose()
    sheetData.columns = ['disease IDs', 'disease ID labels', 'list of diseases', 'active ingredients in therapy', 'drug ID', 'drug ID Label']
    return sheetData
    #print(sheetData)
    #sheetData.to_excel("indication-list-pmda-v1.xlsx")