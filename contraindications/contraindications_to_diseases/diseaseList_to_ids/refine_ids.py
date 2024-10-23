# import pandas as pd
# from tqdm import tqdm
# import time
# from io import StringIO
# import requests

# def getCurie(name, biolink_class):
#     """
#     Args:
#         name (str): string to be identified
#         params (tuple): name resolver parameters to feed into get request
    
#     Returns:
#         resolvedName (list[str]): IDs most closely matching string.
#         resolvedLabel (list[str]): List of labels associated with respective resolvedName.

#     """
#     #return [name], [name] #only for testing
#     url = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=true&offset=0&limit=10&biolink_type=ChemicalOrDrugOrTreatment'
#     success = False
#     failedCounts = 0
#     while not success:
#         try:
#             returned = (pd.read_json(StringIO(requests.get(url).text)))
#             resolvedName = returned.curie
#             resolvedLabel = returned.label
#             success = True
#         except:
#             print(f'name resolver error number {failedCounts}')
#             failedCounts += 1
        
#         if failedCounts >= 10:
#             return "Error", "Error"
#     return resolvedName, resolvedLabel


# current_list = pd.read_excel("contraindicationList_partial.xlsx")
# for idx, row in tqdm(current_list.iterrows(), total=len(current_list)):
#     if row['drug ID']=="Error":
#         new_id, new_label = getCurie(row['active ingredients'], "ChemicalOrDrugOrTreatment")
#         print(f"New IDs: {new_id[0]}, {new_label[0]}")
#         row['drug ID'] = new_id[0]
#         row['drug label'] = new_label[0]


# current_list.to_excel("contraindication_list_filled.xlsx")



import pandas as pd
from tqdm import tqdm
import time
from io import StringIO
import requests

# Global cache dictionary to store previous results
name_cache = {}

def getCurie(name, biolink_class):
    """
    Args:
        name (str): string to be identified
        biolink_class (str): biolink class parameter for the query
    
    Returns:
        resolvedName (list[str]): IDs most closely matching string.
        resolvedLabel (list[str]): List of labels associated with respective resolvedName.
    """
    # Check if result is already in cache
    if name in name_cache:
        return name_cache[name]
    
    url = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=true&offset=0&limit=10&biolink_type=ChemicalOrDrugOrTreatment'
    success = False
    failedCounts = 0
    
    while not success:
        try:
            returned = pd.read_json(StringIO(requests.get(url).text))
            resolvedName = returned.curie
            resolvedLabel = returned.label
            success = True
            
            # Store result in cache
            name_cache[name] = (resolvedName, resolvedLabel)
            
        except:
            print(f'name resolver error number {failedCounts}')
            failedCounts += 1
        
        if failedCounts >= 10:
            error_result = ("Error", "Error")
            name_cache[name] = error_result  # Cache error results too
            return error_result
            
    return resolvedName, resolvedLabel

# Read and process the Excel file
current_list = pd.read_excel("contraindicationList_partial.xlsx")

for idx, row in tqdm(current_list.iterrows(), total=len(current_list)):
    if row['drug ID'] == "Error":
        new_id, new_label = getCurie(row['active ingredients'], "ChemicalOrDrugOrTreatment")
        #print(f"New IDs: {new_id[0]}, {new_label[0]}")
        current_list.at[idx,'drug ID'] = new_id[0]
        current_list.at[idx,'drug label'] = new_label[0]

current_list.to_excel("contraindication_list_filled.xlsx")

# Optional: Print cache statistics at the end
print(f"Cache size: {len(name_cache)} entries")
print(f"Unique queries made")