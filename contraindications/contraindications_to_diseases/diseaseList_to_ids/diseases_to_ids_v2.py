import aiohttp
import asyncio
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import tqdm.asyncio

@retry(wait=wait_exponential(multiplier=1, max=4))
async def fetch_with_progress(session, name, biolink_type):
    url = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_type}'
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()

async def main(names, biolink_class):
    async with aiohttp.ClientSession() as session:
        pbar = tqdm.asyncio(total=len(names), desc="Processing requests")
        tasks = [tqdm.asyncio.fetch_with_progress(session, name, biolink_class, pbar) for name in names]
        responses = await asyncio.gather(*tasks)
        pbar.close()
    return responses

async def fetch_with_progress(session, name, biolink_class, pbar):
    try:
        result = await fetch(session, name, biolink_class)
    except Exception as e:
        result = e
    pbar.update(1)
    return result

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

names = diseaseList
# Run the async function
drug_class = "ChemicalOrDrugOrTreatment"
responses = asyncio.run(main(names, drug_class))


# Print responses (they will be in the same order as the input)
for name, response in zip(names, responses):
    print(f"Response for {name}:")
    if isinstance(response, Exception):
        print(f"Error: {response}")
    else:
        print(response)
    print()