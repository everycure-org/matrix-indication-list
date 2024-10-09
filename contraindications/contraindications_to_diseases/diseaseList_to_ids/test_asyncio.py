import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm
import pandas as pd

response_cache = {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch(session, name, biolink_class):
    url = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_class}'
    async with session.get(url) as response:
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
        result = e 
    pbar.update(1)
    return result

async def fetch_all(names, biolink_class):
    cache = {}
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(names), desc="Processing requests")
        tasks = [fetch_with_cache_and_progress(session, name, biolink_class, pbar) for name in names]
        responses = await asyncio.gather(*tasks)
        pbar.close()
    return responses

def get_curies_and_labels(response):
    try: 
        df = pd.DataFrame.from_dict(response)
        try:
            print(df.curie[0])
            print(df.label[0])
            return df.curie[0], df.label[0]
        except:
            return "Error", "Error"
    except:
        print(f"error reading in JSON for {str(response)}" )
        return "Error", "Error"

diseaseData = pd.read_excel('../active_ingredients_to_structured_lists_v2.xlsx')
diseaseList = []
drugList = []
source_list = []

print("creating tasks")
n_sections = len(diseaseData)
index_min = 0
#limit = 100
limit = n_sections
for index, row in diseaseData.iterrows():
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

biolink_class_drug = "ChemicalOrDrugOrTreatment"
biolink_class_disease = "DiseaseOrPhenotypicFeature"

print("Resolving Drug IDs")
responses_drug = asyncio.run(fetch_all(drugList, biolink_class_drug))
print("Resolving Disease IDs")
responses_disease = asyncio.run(fetch_all(diseaseList, biolink_class_disease))

drug_curie_label_list = list(get_curies_and_labels(r) for r in responses_drug)
disease_curie_label_list = list(get_curies_and_labels(r) for r in responses_disease)


#print(drug_curie_label_list)
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
    "source list": source_list
})

data.to_excel("contraindicationList.xlsx")