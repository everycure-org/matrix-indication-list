import pandas as pd
from io import StringIO
import requests
import tqdm
import tqdm.asyncio
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits
import time
#from joblib import Memory
#memory = Memory(location=".cache/nameres", verbose=0)

class TooManyRequestsError(Exception):
    pass

async def handle_429(response):
    retry_after = response.headers.get('Retry-After')
    if retry_after:
        wait_time = int(retry_after)
    else:
        wait_time = 60  # Default to 60 seconds if no Retry-After header
    print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying.")
    await asyncio.sleep(wait_time)
    raise TooManyRequestsError("Rate limit exceeded")

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=15))
async def getCurie_async(session, name, disease_id_label_list, index, cache, biolink_class):
    try:
        if name in cache:
            disease_id_label_list[index] = cache[name]
        else:
            itemRequest = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_class}'
            async with session.get(itemRequest, timeout=30) as response:
                if response.status == 429:
                    await handle_429(response)
                out_json = await response.json()
            
            try:
                returned = pd.DataFrame.from_dict(out_json)
                resolvedName = returned.curie
                resolvedLabel = returned.label
                disease_id_label_list[index] = resolvedName[0], resolvedLabel[0]
                cache.update({name: (resolvedName[0], resolvedLabel[0])})
            except pd.errors.EmptyDataError:
                print(f"Empty response for {name}")
                disease_id_label_list[index] = "Error", "Empty Response"
    except TooManyRequestsError:
        raise
    except asyncio.CancelledError:
        print(f"Task for {name} was cancelled")
        disease_id_label_list[index] = "Error", "Task Cancelled"
    except tenacity.RetryError as e:
        if isinstance(e.last_attempt._exception, asyncio.TimeoutError):
            print(f"Timeout error after retries for {name}")
            disease_id_label_list[index] = "Error", "Timeout After Retries"
        else:
            print(f"Retry error for {name}: {str(e)}")
            disease_id_label_list[index] = "Error", f"Retry Error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error processing {name}: {str(e)}")
        disease_id_label_list[index] = "Error", f"Unexpected Error: {str(e)}"
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        print(f"Network error processing {name}: {str(e)}")
        disease_id_label_list[index] = "Error", f"Network Error: {str(e)}"
    finally:
        await asyncio.sleep(0.1)  # Add a small delay to avoid overwhelming the server



# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def getCurie_async(session, name, disease_id_label_list, index, cache, biolink_class):
#     if name in cache:
#         disease_id_label_list[index] = cache[name]
#     else:
#         itemRequest = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_class}'
#         async with session.get(itemRequest, timeout=30) as response:
#             if response.status == 429:
#                 await handle_429(response)
#             #response.raise_for_status()
#             out_json = await response.json()
        
#         try:
#             returned = pd.DataFrame.from_dict(out_json)
#             resolvedName = returned.curie
#             resolvedLabel = returned.label
#             disease_id_label_list[index] = resolvedName[0], resolvedLabel[0]
#             cache.update({name: (resolvedName[0], resolvedLabel[0])})
#         except TooManyRequestsError:
#             raise
#         except asyncio.CancelledError:
#             print(f"Task for {name} was cancelled")
#             disease_id_label_list[index] = "Error", "Task Cancelled"
#         except tenacity.RetryError as e:
#             if isinstance(e.last_attempt._exception, asyncio.TimeoutError):
#                 print(f"Timeout error after retries for {name}")
#                 disease_id_label_list[index] = "Error", "Timeout After Retries"
#             else:
#                 print(f"Retry error for {name}: {str(e)}")
#                 disease_id_label_list[index] = "Error", f"Retry Error: {str(e)}"
#         except (asyncio.TimeoutError, aiohttp.ClientError, pd.errors.EmptyDataError) as e:
            
#             time.sleep(.1)
#             print(f"Error processing {name}: {str(e)}")
#             disease_id_label_list[index] = "Error", str(e)
#         except Exception as e:
#             time.sleep(.1)
#             print(f"Unexpected error processing {name}: {str(e)}")
#             disease_id_label_list[index] = "Error", "Unexpected Error"
#             print(f"Response:{str(out_json)}")

def build_string_from_list(list):
    outString = "["
    for item in list:
        outString += item + ", "
    outString = outString[:-2] + "]"
    return outString

async def getAllCuries(names: list[str], ids_preallocated, biolink_type, concurrency=5):
    cache = {}
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        async def bounded_fetch(name, idx):
            async with semaphore:
                await getCurie_async(session, name, ids_preallocated, idx, cache, biolink_type)
        
        tasks = [bounded_fetch(name, idx) for idx, name in enumerate(names)]
        print("Resolving IDs and labels...")
        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            await f

async def getAllCuries(names: list[str], ids_preallocated, biolink_type):
    cache = {}
    async with aiohttp.ClientSession() as session:
        tasks = [getCurie_async(session, name, ids_preallocated, idx, cache, biolink_type) for idx, name in enumerate(names)]
        print("Resolving IDs and labels...")
        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            await f
        
async def main():
    diseaseData = pd.read_excel('../active_ingredients_to_structured_lists_v2.xlsx')
    diseaseList = []
    drugList = []
    source_list = []

    print("creating tasks")
    n_sections = len(diseaseData)
    index_min = 0
    #limit = n_sections
    limit = 5000
    for index, row in tqdm.tqdm(diseaseData.iterrows(), total=n_sections):
        drug = row['Active Ingredients']
        diseases = row['Structured Disease list']
        if index >= index_min and index < limit:
            curr_row_diseasesTreated = row['Structured Disease list']    
            if type(curr_row_diseasesTreated)!=float:
                curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').split(',')
                for idx2,item in enumerate(curr_row_diseaseList):             
                    item = item.strip().upper().replace(" \n","").replace(" (PREVENTATIVE)","")
                    diseaseList.append(item)
                    drugList.append(drug)
                    source_list.append(diseases)

    disease_ids_and_labels = [("","")]*len(diseaseList)
    drug_ids_and_labels = [("","")]*len(diseaseList)
    print("Resolving disease IDs")
    await getAllCuries(diseaseList, disease_ids_and_labels, 'DiseaseOrPhenotypicFeature')
    
    print("Resolving drug IDs")
    await getAllCuries(drugList, drug_ids_and_labels, 'ChemicalOrDrugOrTreatment')
    diseaseIDList, diseaseLabelList = zip(*disease_ids_and_labels)
    drugIDList, drugLabelList = zip(*drug_ids_and_labels)

    data = pd.DataFrame({
        "active ingredients": drugList,
        "drug ID": drugIDList,
        "drug label": drugLabelList,
        "disease list": diseaseList,
        "disease curie": diseaseIDList,
        "disease label": diseaseLabelList,
        "source list": source_list
    })
    data.to_excel("contraindicationList.xlsx")

asyncio.run(main())
print("finished building contraindications list.")