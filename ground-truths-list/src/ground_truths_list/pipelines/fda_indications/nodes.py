"""
This is a boilerplate pipeline 'fda_indications'
generated using Kedro 0.19.9
"""
import pandas as pd
from tqdm import tqdm
import pandas as pd
from io import StringIO
import requests

import pandas as pd 
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import vertexai.preview.generative_models as generative_models

testing = True
limit = 100

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


def extract_fda_indications(inputList: pd.DataFrame, ) -> pd.DataFrame:
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
                input_text = f"Produce a list of diseases treated in the following therapeutic indications list:\n {item} \n Please format the list as: \'item1|item2|...|itemN\'. Do not include any other text in the response. If no diseases are treated, return \'None\'. If the drug is only used for diagnostic or procedural purposes, return \'non-therapeutic\'."
                response = generate(input_text, safety_settings, generation_config)
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

    labelDict = {}
    idDict = {}

    for index, row in tqdm(diseaseData.iterrows(), total=len(diseaseData)):
        curr_row_diseasesTreated = row['diseases treated']
        if type(curr_row_diseasesTreated)!=float:
            
            curr_row_drugsInTherapy = row['active ingredient(s)']
            curr_row_disease_ids = []
            curr_row_disease_id_labels = []
            curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').split(',')
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
                item = item.strip().upper().replace(" \n","").replace(" (PREVENTATIVE)","")
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
                    
                except:
                    print("error during name resolving")

    sheetData = pd.DataFrame(data=[diseaseIDList, diseaseLabelList, diseaseList, drugList, drugIDList, drugLabelList]).transpose()
    sheetData.columns = ['disease IDs', 'disease ID labels', 'list of diseases', 'active ingredients in therapy', 'drug ID', 'drug ID Label']
    return sheetData