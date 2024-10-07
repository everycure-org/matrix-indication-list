import pandas as pd
from io import StringIO
import requests
import tqdm
import tqdm.asyncio
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#@limits(calls=500, period=1)
async def getCurie_Disease_async(session, name, disease_id_label_list, index, cache):
    if name in cache:
        disease_id_label_list[index] = cache[name]
        return
    else:
        itemRequest = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type=DiseaseOrPhenotypicFeature'
        async with session.get(itemRequest) as response:
            text = await response.text()
        try:
            returned = pd.read_json(StringIO(text))
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#@limits(calls=500, period=1)
async def getCurie_Drug_async(session, name, drug_id_label_list, index, cache):
    if name in cache:
        #print(f"cache hit for {name}")
        drug_id_label_list[index] = cache[name]
        return
    else:
        itemRequest = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type=ChemicalOrDrugOrTreatment'
        async with session.get(itemRequest) as response:
            text = await response.text()
        try:
            returned = pd.read_json(StringIO(text))
            resolvedName = returned.curie
            resolvedLabel = returned.label
            drug_id_label_list[index] = resolvedName[0], resolvedLabel[0]
            cache.update({name: (resolvedName[0], resolvedLabel[0])})
        except (asyncio.TimeoutError, aiohttp.ClientError, pd.errors.EmptyDataError) as e:
            print(f"Error processing {name}: {str(e)}")
            drug_id_label_list[index] = "Error", str(e)
        except Exception as e:
            print(f"Unexpected error processing {name}: {str(e)}")
            drug_id_label_list[index] = "Error", "Unexpected Error"

def build_string_from_list(list):
    outString = "["
    for item in list:
        outString += item + ", "
    outString = outString[:-2] + "]"
    return outString

async def getAllDiseaseCuries(names: list[str], disease_ids_preallocated):
    cache = {}
    async with aiohttp.ClientSession() as session:
        tasks = [getCurie_Disease_async(session, name, disease_ids_preallocated, idx, cache) for idx, name in enumerate(names)]
        print("Resolving disease IDs and labels...")
        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            await f

async def getAllDrugCuries(names: list[str], drug_ids_preallocated):
    cache = {}
    async with aiohttp.ClientSession() as session:
        tasks = [getCurie_Drug_async(session, name, drug_ids_preallocated, idx, cache) for idx, name in enumerate(names)]
        print("Resolving drug IDs and labels...")
        for f in tqdm.asyncio.tqdm.as_completed(tasks):
            try:
                await f
            except:
                print("exception at await task level")
        
async def main():
    diseaseData = pd.read_excel('../active_ingredients_to_structured_lists_v2.xlsx')
    diseaseList = []
    drugList = []
    source_list = []
    
    nRows = len(list(diseaseData['Structured Disease list']))
    
    labelDict = {}
    idDict = {}
        
    
    print("creating tasks")
    n_sections = len(diseaseData)
    #limit = n_sections
    limit = 10000
    for index, row in tqdm.tqdm(diseaseData.iterrows(), total=n_sections):
        drug = row['Active Ingredients']
        diseases = row['Structured Disease list']
        if index < limit:
            curr_row_diseasesTreated = row['Structured Disease list']
            
            if type(curr_row_diseasesTreated)!=float:
                curr_row_drugsInTherapy = row['Active Ingredients']
                curr_row_disease_ids = []
                curr_row_disease_id_labels = []
                curr_row_diseaseList = curr_row_diseasesTreated.replace("[","").replace("]","").replace('\'','').split(',')
        
                for idx2,item in enumerate(curr_row_diseaseList):             
                    item = item.strip().upper().replace(" \n","").replace(" (PREVENTATIVE)","")
                    diseaseList.append(item)
                    drugList.append(drug)
                    source_list.append(diseases)

    print(len(diseaseList), " items in disease list")
    print(f"Correspondingly, {len(drugList)} sets of active ingredients")

    disease_ids_and_labels = [("","")]*len(diseaseList)
    drug_ids_and_labels = [("","")]*len(diseaseList)

    await getAllDiseaseCuries(diseaseList, disease_ids_and_labels)
    await getAllDrugCuries(drugList, drug_ids_and_labels)

    print(disease_ids_and_labels)
    print(drug_ids_and_labels)

    print(type(disease_ids_and_labels))
    print(type(drug_ids_and_labels))

    diseaseIDList, diseaseLabelList = zip(*disease_ids_and_labels)
    drugIDList, drugLabelList = zip(*drug_ids_and_labels)
    
    print(len(drugList))
    print(len(drugIDList))
    print(len(drugLabelList))
    print(len(diseaseList))
    print(len(diseaseIDList))
    print(len(diseaseLabelList))
    print(len(source_list))


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