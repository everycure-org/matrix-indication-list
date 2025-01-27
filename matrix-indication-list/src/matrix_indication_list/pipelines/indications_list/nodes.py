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

## Testing Params (set to true to limit number of labels parsed)
testing = False
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


def prompt_llm(prompt: str, params: dict) -> str:

    return "NONE"


def extract_named_diseases(inList:pd.DataFrame, column_names: dict, gemini_generation_config: dict) -> pd.DataFrame:
    indicationsData = list(inList[column_names.get("indications_text_column")])
    activeIngredientsData = list(inList[column_names.get("drug_name_column")])
    print(len(indicationsData), 'indications sections found in list')
    
    diseasesTreated = []
    therapyActiveIngredients = []
    originalText = []
    cache = {}
    for index, item in tqdm(enumerate(indicationsData), total=(limit if testing else len(indicationsData))):
        if (testing and index < limit) or not testing:
            try:
                curr_prompt = prompt + item
                if curr_prompt in cache:
                    response = cache[curr_prompt]
                else:
                    response = generate(curr_prompt, safety_settings, gemini_generation_config)
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

def flatten_list (inList: pd.DataFrame) -> pd.DataFrame:
    
    return None

def resolve_drugs (inList: pd.DataFrame, params:dict) -> pd.DataFrame:

    return None

def resolve_diseases (inList: pd.DataFrame, params:dict) -> pd.DataFrame:

    return None