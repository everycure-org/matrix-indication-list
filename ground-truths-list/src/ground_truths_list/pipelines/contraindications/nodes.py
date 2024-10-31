"""
This is a boilerplate pipeline 'contraindications'
generated using Kedro 0.19.9
"""

import pandas as pd
import asyncio
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from typing import List, Any
from dataclasses import dataclass
import time
from tqdm import tqdm
import math
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import tqdm.asyncio
import numpy as np

# Set your Google Cloud project ID and location
PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
LOCATION = "us-central1"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

testing = False
limit = 100

@dataclass
class BatchResult:
    index: int
    response: str

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()

async def predict_gemini_model(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    content: str,
    rate_limiter: RateLimiter,
) -> str:
    """Predict using a Gemini Model with safety settings."""
    await rate_limiter.acquire()
    
    model = GenerativeModel(model_name)
    
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
    }
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            contents=content,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    return part.text.strip()
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "ERROR: API call failed"
        
    return "NO LLM OUTPUT"

async def process_batch(
    prompts: List[str],
    start_index: int,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    model_name: str = "gemini-1.5-flash-001"
) -> List[BatchResult]:
    """Process a batch of prompts while maintaining their original order."""
    tasks = []
    
    for i, prompt in enumerate(prompts):
        async def process_single(prompt: str, index: int) -> BatchResult:
            async with semaphore:
                response = await predict_gemini_model(
                    model_name=model_name,
                    temperature=0.2,
                    max_output_tokens=1024,
                    top_p=0.8,
                    top_k=40,
                    content=prompt,
                    rate_limiter=rate_limiter,
                )
                return BatchResult(index=start_index + index, response=response)
                
        tasks.append(process_single(prompt, i))
    
    results = await asyncio.gather(*tasks)
    return results

async def process_all_prompts(
    prompts: List[str],
    batch_size: int = 50,
    max_concurrent: int = 10,
    requests_per_minute: int = 200,
) -> List[str]:
    """Process all prompts in batches with concurrency control and rate limiting."""
    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limiter = RateLimiter(requests_per_minute)
    
    # Calculate total number of batches for tqdm
    total_batches = math.ceil(len(prompts) / batch_size)
    
    # Create progress bar for batches
    with tqdm.tqdm(total=len(prompts), desc="Processing prompts") as pbar:
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            batch_results = await process_batch(
                prompts=batch,
                start_index=i,
                semaphore=semaphore,
                rate_limiter=rate_limiter
            )
            
            all_results.extend(batch_results)
            pbar.update(len(batch))
    
    # Sort results by original index and extract responses
    all_results.sort(key=lambda x: x.index)
    return [result.response for result in all_results]

def get_input_text(active_ingredient_data, contraindication_text):
    text = "Produce a list of diseases contraindicated for the active ingredient " + str(active_ingredient_data) + " based on the following contraindications list:\n" + str(contraindication_text) + "Please format the list as [\'item1\', \'item2\', ... ,\'itemN\']. Do not include any other text in the response. If no diseases are contraindicated for, return an empty list as \'[]\'. If the drug is only used for diagnostic purposes, return \'diagnostic/contrast/radiolabel\'. Do not include hypersensitivity or allergy to the named drug as a contraindication. This code is being deployed in bulk so if the contraindications section is just \'template\' or similar, return an empty list. Only include conditions that the drug causes, and omit pre-exisitng conditions of the patients."
    return text   

def generate_prompts(contraindications_data, active_ingredients_data, pLimit) -> List[str]:
    print("Generating prompts...")
    prompts = []
    n_contraindications = min(len(contraindications_data), pLimit)
    print(f"generating {n_contraindications} prompts")
    for index in range(n_contraindications):
        prompts.append(get_input_text(active_ingredients_data[index], contraindications_data[index]))
    print(f"generated {len(prompts)} prompts")
    return prompts

async def llm_extract_contraindications(data_in: pd.DataFrame) -> pd.DataFrame:
    # Configuration
    BATCH_SIZE = 50  # Number of prompts to process in each batch
    MAX_CONCURRENT = 10  # Maximum number of concurrent API calls
    REQUESTS_PER_MINUTE = 200  # Maximum number of API requests per minute
    
    # Load data
    drugs_to_contraindications = data_in
    contraindications_data = list(drugs_to_contraindications['contraindications'])
    active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

    #limit = len(contraindications_data)
    prompts = generate_prompts(contraindications_data, active_ingredients_data, pLimit=len(contraindications_data))
    
    print(f"Starting processing of {len(prompts)} prompts:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max concurrent calls: {MAX_CONCURRENT}")
    print(f"- Rate limit: {REQUESTS_PER_MINUTE} requests/minute")
    
    start_time = time.time()
    
    responses = await process_all_prompts(
        prompts=prompts,
        batch_size=BATCH_SIZE,
        max_concurrent=MAX_CONCURRENT,
        requests_per_minute=REQUESTS_PER_MINUTE
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds")

    # Save results
    print(len(active_ingredients_data[0:limit]), len (responses[0:limit]), len(contraindications_data[0:limit]))
    data = pd.DataFrame({
        "Active Ingredients": active_ingredients_data[0:limit],
        "Structured Disease list": responses[0:limit],
        "Source Text": contraindications_data[0:limit],
    })
    return data

def extract_structured_lists_contraindications_dailymed (dailymed_contraindications: pd.DataFrame)->pd.DataFrame:
    data = asyncio.run(llm_extract_contraindications(dailymed_contraindications))
    return data



##############
## STRUCTURED LISTS TO IDS

response_cache = {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch(session, name, biolink_class):
    url = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_class}'
    async with session.get(url, timeout=30) as response:
        response.raise_for_status()
        return await response.json()

async def fetch_with_cache_and_progress(session, name, biolink_class, pbar):
    cache_key = (name, biolink_class)
    if cache_key in response_cache:
        pbar.update(1)
        return response_cache[cache_key]
    try:
        result = await fetch(session, name, biolink_class)
        response_cache[cache_key] = result
    except Exception as e:
        result = str(e)
    pbar.update(1)
    return result

async def fetch_all(names, biolink_class, batch_size=1000):
    results = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        async with aiohttp.ClientSession() as session:
            pbar = tqdm.tqdm(total=len(batch), desc=f"Processing batch {i//batch_size + 1}")
            tasks = [fetch_with_cache_and_progress(session, name, biolink_class, pbar) for name in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
            pbar.close()
        print(f"Completed batch {i//batch_size + 1}. Sleeping for 10 seconds...")
        await asyncio.sleep(10)  # Add a delay between batches
    return results

def get_curies_and_labels(response):
    if isinstance(response, str):  # This is an error message
        print("error type 1")
        print(str(response))
        return "Error", "Error"
    try: 
        df = pd.DataFrame.from_dict(response)
        try:
            return df.curie.iloc[0], df.label.iloc[0]
        except:
            print("exception raised when returning result")
            return "Error", "Error"
    except Exception as e:
        print(f"error reading in JSON for {str(response)}: {e}")
        return "Error", "Error"


def generate_contraindications_list(diseaseData: pd.DataFrame)-> pd.DataFrame:
    diseaseList = []
    drugList = []
    source_list = []
    source_text = []

    print("creating tasks")
    n_sections = len(diseaseData)
    for index, row in diseaseData.iterrows():
        drug = row['Active Ingredients']
        diseases = row['Structured Disease list']
        src = row['Source Text']
        if not testing or index < limit:
            curr_row_diseasesTreated = row['Structured Disease list']    
            if type(curr_row_diseasesTreated)!=float:
                curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').split(',')
                for idx2,item in enumerate(curr_row_diseaseList):             
                    item = item.strip().upper().replace(" \n","").replace(" (PREVENTATIVE)","")
                    diseaseList.append(item)
                    drugList.append(drug)
                    source_list.append(diseases)
                    source_text.append(src)

    biolink_class_drug = "ChemicalOrDrugOrTreatment"
    biolink_class_disease = "DiseaseOrPhenotypicFeature"

    print("Resolving Drug IDs")
    responses_drug = asyncio.run(fetch_all(drugList, biolink_class_drug))
    print("Resolving Disease IDs")
    responses_disease = asyncio.run(fetch_all(diseaseList, biolink_class_disease))

    drug_curie_label_list = list(get_curies_and_labels(r) for r in responses_drug)
    disease_curie_label_list = list(get_curies_and_labels(r) for r in responses_disease)
    #print(disease_curie_label_list)

    disease_IDs, disease_labels = zip(*disease_curie_label_list)
    drug_IDs, drug_labels = zip(*drug_curie_label_list)

    data = pd.DataFrame({
        "active ingredients": drugList,
        "drug ID": drug_IDs,
        "drug label": drug_labels,
        "disease list": diseaseList,
        "disease curie": disease_IDs,
        "disease label": disease_labels,
        "source list": source_list,
        "original source text": source_text,
    })
    return data


def merge_contraindications_and_indications(contraindications: pd.DataFrame, indications: pd.DataFrame) -> pd.DataFrame:
    #print("reading files...")
    #indications = pd.read_csv("../merge_lists/indicationList.tsv", sep='\t')
    #contraindications = pd.read_excel("../contraindications/contraindications_to_diseases/diseaseList_to_ids/contraindication_list_filled.xlsx")

    print("removing unneeded columns and dropping duplicate entries")
    #contraindications.drop('Unnamed: 0.2', axis=1, inplace=True)
    #contraindications.drop('Unnamed: 0.1', axis=1, inplace=True)
    #contraindications.drop('Unnamed: 0', axis=1, inplace=True)
    contraindications.drop_duplicates(subset=['active ingredients', 'drug ID', 'disease curie'], keep='first')

    print("tagging contraindications and indications")
    contraindications['contraindication'] = True
    indications['indication'] = True
    contraindications = contraindications.rename(columns={'disease list': 'disease name', 'disease curie':'disease ID'})
    indications = indications.rename(columns={'disease list':'disease name', 
                                            'disease curie':'disease ID', 
                                            'active ingredients in therapy':'active ingredients',
                                            'disease ID labels':'disease label',
                                            'drug ID Label': 'drug label', 
                                            'disease IDs': 'disease ID',
                                            'list of diseases': 'disease name',
                                            })
    contraindications['drug|disease'] = list(f"{row['drug ID']}|{row['disease ID']}" for idx,row in contraindications.iterrows())
    indications = indications[['active ingredients', 
                            'drug ID', 
                            'drug label', 
                            'disease name', 
                            'disease ID', 
                            'disease label',  
                            'indication', 
                            'drug|disease',
                            ]]
    print("combining lists...")
    ground_truths_list = pd.concat([indications, contraindications], axis=1)
    result = pd.concat([indications, contraindications], axis=0).reset_index(drop=True)
    result.drop('source list', axis=1, inplace=True)

    print("adding indication / contraindication tags to drugs...")
    for idx, row in tqdm.tqdm(result.iterrows(), total = len(result)):
        if np.isnan((row['indication'])):
            result.loc[idx, "indication"] = False
            #result['indication'][idx]=False
        if np.isnan((row['contraindication'])):
            #result['contraindication'][idx]=False
            result.loc[idx, "contraindication"] = False
    #print("saving file...")
    return result
    #result.to_csv("ground-truths-list.tsv", sep='\t')
    #print("completed indication and contraindication merging")