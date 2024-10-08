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

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
async def getCurie_async(session, name, disease_id_label_list, index, cache, biolink_class):
    if name in cache:
        disease_id_label_list[index] = cache[name]
    else:
        itemRequest = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_class}'
        async with session.get(itemRequest) as response:
            out_json = await response.json()
        try:
            returned = pd.DataFrame.from_dict(out_json)
            resolvedName = returned.curie
            resolvedLabel = returned.label
            disease_id_label_list[index] = resolvedName[0], resolvedLabel[0]
            cache.update({name: (resolvedName[0], resolvedLabel[0])}) 
        except (asyncio.TimeoutError, aiohttp.ClientError, pd.errors.EmptyDataError) as e:
            print(f"Error processing {name}: {str(e)}")
            disease_id_label_list[index] = "Error", str(e)
        except Exception as e:
            print(f"Unexpected error processing {name}: {str(e)}")
            disease_id_label_list[index] = "Error", "Unexpected Error"
            print(f"Response:{response}")

def build_string_from_list(list):
    outString = "["
    for item in list:
        outString += item + ", "
    outString = outString[:-2] + "]"
    return outString

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