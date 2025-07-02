import xml.etree.ElementTree as ET
from tqdm import tqdm
import os 
import pandas as pd
import zipfile
import re


def mine_indications(dir: str) -> pd.DataFrame:
    indications_list = []
    ingredients_list = []
    counts = 0
    foundCounts = 0
    notFoundCounts = 0
    dirs = []
    # TODO: automatically find, download, unzip all of the dailymed folders
    labelFolders = ["prescription_1/", "prescription_2/", "prescription_3/", "prescription_4/", "prescription_5/"]
    
    for label in labelFolders:
        dirs.append(dir+label)

    for directory in (dirs):
        for files in tqdm(os.listdir(directory), desc=f"reading directory {directory}"):
            if files.endswith(".zip"):
                fpath = directory + files
                fileRoot = files.replace(".zip","")
                dest = directory + fileRoot
                try:
                    unzip_file(fpath,dest)
                except:
                    #print("failed to unzip file ", fpath)
                    continue
                xmlfile=""
                for contents in os.listdir(dest):
                    if contents.endswith(".xml"):
                        xmlfile=contents.replace("._","")
                xmlfilepath = dest+"/"+xmlfile
                indications = extract_indications(xmlfilepath)
                active_ingredients = extract_active_ingredient(xmlfilepath)
                for ind, item in enumerate(active_ingredients):
                    active_ingredients[ind]=item.upper()
                ingredients_list.append(set(active_ingredients))
                if indications is not None:
                    indications_list.append(indications['indications'])
                    foundCounts += 1
                    #print(foundCounts, " indications successfully found so far")
                else:
                    notFoundCounts += 1
                    #print(notFoundCounts, " indications not found so far, failed to find for ", files)
                    indications_list.append("")
                counts +=1
        
    print("finished ingesting indications")
    data = pd.DataFrame({'active ingredient':ingredients_list, 'indications':indications_list})
    return data


def mine_usage(dir: str) -> pd.DataFrame:
    ingredients_list = []
    usage_list = []
    counts = 0
    foundCounts = 0
    notFoundCounts = 0
    dirs = []
    # TODO: automatically find, download, unzip all of the dailymed folders
    labelFolders = ["prescription_1/", "prescription_2/", "prescription_3/", "prescription_4/", "prescription_5/"]
    
    for label in labelFolders:
        dirs.append(dir+label)

    for directory in (dirs):
        for files in tqdm(os.listdir(directory), desc=f"reading directory {directory}"):
            if files.endswith(".zip"):
                fpath = directory + files
                fileRoot = files.replace(".zip","")
                dest = directory + fileRoot
                try:
                    unzip_file(fpath,dest)
                except:
                    #print("failed to unzip file ", fpath)
                    continue
                xmlfile=""
                for contents in os.listdir(dest):
                    if contents.endswith(".xml"):
                        xmlfile=contents.replace("._","")
                xmlfilepath = dest+"/"+xmlfile
                usage = extract_usage(xmlfilepath)
                active_ingredients = extract_active_ingredient(xmlfilepath)
                for ind, item in enumerate(active_ingredients):
                    active_ingredients[ind]=item.upper()
                ingredients_list.append(set(active_ingredients))
                if usage is not None:
                    usage_list.append(usage['usage'])
                    foundCounts += 1
                    #print(foundCounts, " indications successfully found so far")
                else:
                    notFoundCounts += 1
                    #print(notFoundCounts, " indications not found so far, failed to find for ", files)
                    usage_list.append("")
                counts +=1
        
    print("finished ingesting indications")
    data = pd.DataFrame({
        'active ingredient':ingredients_list, 
        'indications':usage_list
        })
    return data

def extract_usage(xml_file_path):
    """
    Extract indications from a Structured Product Label XML file.
    
    Args:
        xml_file_path (str): Path to the SPL XML file
        
    Returns:
        dict: Dictionary containing indications text and metadata
    """
    # Define the namespace used in SPL files
    namespaces = {
        'v3': 'urn:hl7-org:v3',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Initialize results dictionary
        results = {
            'usage': None,
            'section_code': None,
            'section_id': None,
            'metadata': {}
        }
        
        # Find the indications section
        # SPL uses code 34070-3 for indications
        for section in root.findall('.//v3:section', namespaces):
            code_element = section.find('.//v3:code', namespaces)
            if code_element is not None:
                code = code_element.get('code')
                if code == '34068-7':  # LOINC code for usage
                    # Extract section ID if present
                    results['section_id'] = section.get('ID', '')
                    
                    # Extract section code details
                    results['section_code'] = {
                        'code': code,
                        'codeSystem': code_element.get('codeSystem', ''),
                        'displayName': code_element.get('displayName', '')
                    }
                    
                    # Extract the text content
                    text_element = section.find('.//v3:text', namespaces)
                    if text_element is not None:
                        # Convert the text element to string, preserving internal markup
                        text_content = ET.tostring(text_element, encoding='unicode', method='xml')
                        
                        # Clean up the text content
                        # Remove XML tags while preserving content
                        clean_text = re.sub(r'<[^>]+>', ' ', text_content)
                        # Remove extra whitespace
                        clean_text = ' '.join(clean_text.split())
                        
                        results['usage'] = clean_text
                    
                    # Extract any additional metadata
                    title_element = section.find('.//v3:title', namespaces)
                    if title_element is not None:
                        results['metadata']['title'] = title_element.text
                    
                    break  # Stop after finding the indications section
        
        return results
    
    except ET.ParseError as e:
        raise Exception(f"Error parsing XML file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing SPL file: {str(e)}")



def extract_indications(xml_file_path):
    """
    Extract indications from a Structured Product Label XML file.
    
    Args:
        xml_file_path (str): Path to the SPL XML file
        
    Returns:
        dict: Dictionary containing indications text and metadata
    """
    # Define the namespace used in SPL files
    namespaces = {
        'v3': 'urn:hl7-org:v3',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Initialize results dictionary
        results = {
            'indications': None,
            'section_code': None,
            'section_id': None,
            'metadata': {}
        }
        
        # Find the indications section
        # SPL uses code 34070-3 for indications
        for section in root.findall('.//v3:section', namespaces):
            code_element = section.find('.//v3:code', namespaces)
            if code_element is not None:
                code = code_element.get('code')
                if code == '34067-9':  # LOINC code for indications
                    # Extract section ID if present
                    results['section_id'] = section.get('ID', '')
                    
                    # Extract section code details
                    results['section_code'] = {
                        'code': code,
                        'codeSystem': code_element.get('codeSystem', ''),
                        'displayName': code_element.get('displayName', '')
                    }
                    
                    # Extract the text content
                    text_element = section.find('.//v3:text', namespaces)
                    if text_element is not None:
                        # Convert the text element to string, preserving internal markup
                        text_content = ET.tostring(text_element, encoding='unicode', method='xml')
                        
                        # Clean up the text content
                        # Remove XML tags while preserving content
                        clean_text = re.sub(r'<[^>]+>', ' ', text_content)
                        # Remove extra whitespace
                        clean_text = ' '.join(clean_text.split())
                        
                        results['indications'] = clean_text
                    
                    # Extract any additional metadata
                    title_element = section.find('.//v3:title', namespaces)
                    if title_element is not None:
                        results['metadata']['title'] = title_element.text
                    
                    break  # Stop after finding the indications section
        
        return results
    
    except ET.ParseError as e:
        raise Exception(f"Error parsing XML file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing SPL file: {str(e)}")

def unzip_file(zip_path, extract_to_folder):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"The file {zip_path} does not exist.")
    os.makedirs(extract_to_folder, exist_ok=True) 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
       #print(f"Extracted all contents to {extract_to_folder}")

def extract_active_ingredient(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'fda': 'urn:hl7-org:v3'}
    active_ingredients = []
    for ingredient in root.findall(".//fda:activeMoiety/fda:name", ns):
        active_ingredients.append(ingredient.text)
    return active_ingredients

# def getIndications(xmlfilepath):
#     tree = ET.parse(xmlfilepath)
#     root = tree.getroot()
#     ns = {'hl7': 'urn:hl7-org:v3'}
#     sections = root.findall('.//hl7:section', namespaces=ns)
#     for section in sections:
#         codeSection = section.find('.//hl7:code', namespaces=ns)
#         code = codeSection.get('code') if codeSection is not None else "no code"
#         if code == "34067-9":
#             text_elem = section.find('.//hl7:text', namespaces=ns)
#             try:
#                 text_content = ''.join(text_elem.itertext()).strip()
#             except:
#                 print('text_elem was empty')
#                 return ""
#             return strip_spaces(text_content.strip(string.whitespace.replace(" ", "")))
#         else:
#             text_elem = None    
#     return None

# def strip_spaces(myString):
#     _RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
#     _RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")
#     myString = _RE_COMBINE_WHITESPACE.sub(" ", myString)
#     myString = _RE_STRIP_WHITESPACE.sub("", myString)
#     return myString