# import asyncio
# from typing import List
# import vertexai
# from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
# from google.cloud import aiplatform
# from IPython.display import display, HTML
# import pandas as pd
# from tqdm import tqdm
# from tenacity import retry, wait_random_exponential
# from tqdm.asyncio import tqdm_asyncio

# # Initialize Google Cloud project and location
# PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
# my_project = PROJECT_ID
# LOCATION = "us-central1"
# aiplatform.init(project=PROJECT_ID, location=LOCATION)

# generation_config = {
#     "max_output_tokens": 8192,
#     "temperature": 1,
#     "top_p": 0.95,
# }

# safety_settings = [
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
# ]

# @retry(wait=wait_random_exponential(multiplier=1, max=4))
# async def process_single_prompt(prompt: str, project_id: str) -> str:
#     """Process a single prompt with retry logic"""
#     vertexai.init(project=project_id, location="us-central1")
#     model = GenerativeModel("gemini-1.5-pro-001")
    
#     response = await model.generate_content_async(
#         [prompt],
#         generation_config=generation_config,
#         safety_settings=safety_settings,
#         stream=False,
#     )
#     return response.text

# async def process_batch(prompts: List[str], project_id: str, batch_size: int) -> List[str]:
#     """Process prompts in batches while maintaining order"""
#     results = []
#     for i in range(0, len(prompts), batch_size):
#         batch = prompts[i:i + batch_size]
#         batch_tasks = [process_single_prompt(prompt, project_id) for prompt in batch]
#         batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Processing batch {i//batch_size + 1}")
#         results.extend(batch_results)
#     return results

# def get_input_text(active_ingredient_data, contraindication_text):
#     text = "Produce a list of diseases contraindicated for the active ingredient " + str(active_ingredient_data) + " in the following contraindications list:\n" + str(contraindication_text) + "\nPlease format the list as [\'item1\', \'item2\', ... ,\'itemN\']. Do not include any other text in the response. If no diseases are contraindicated for, return an empty list as \'[]\'. This code is being deployed in bulk so if the contraindications section is just \<template\> or similar, return an empty list. If including allergic reactions or anaphylactic reactions, ensure they refer to the specific drug, e.g. \'anaphylactic reactions to solifenacin\' "
#     return text   

# def generate_prompts(contraindications_data, active_ingredients_data, limit) -> List[str]:
#     print("generating prompts...")
#     prompts = []
#     n_contraindications = len(contraindications_data)
#     for index, item in tqdm(enumerate(contraindications_data), total=min(n_contraindications, limit)):
#         if index < limit:
#             prompts.append(get_input_text(active_ingredients_data[index], item))
#     return prompts

# async def main():
#     # Load data
#     drugs_to_contraindications = pd.read_excel("../contraindicationList.xlsx")
#     contraindications_data = list(drugs_to_contraindications['contraindications'])
#     active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

#     # Generate prompts
#     prompts = generate_prompts(contraindications_data, active_ingredients_data, limit=len(contraindications_data))
#     print(f"Found {len(prompts)} prompts to process")

#     # Process all prompts while maintaining order
#     responses = await process_batch(prompts, my_project, batch_size=100)
    
#     # Store structured responses (maintaining order)
#     structured_lists = responses  # responses are already in the correct order

#     # Optional: Display results
#     # display_responses(prompts, responses)
    
#     return active_ingredients_data, responses, structured_lists, contraindications_data

# if __name__ == "__main__":
#     active_ingredients_data, responses, structured_lists, contraindications_text = asyncio.run(main())
#     data = pd.DataFrame({
#         "active ingredients": active_ingredients_data[0:len(structured_lists)],
#         "structured list": structured_lists,
#         "original text": contraindications_text[0:len(structured_lists)]
#     })
#     data.to_excel("indications_to_diseases.xlsx")








# import asyncio
# from typing import List, Dict, Any, Tuple
# import vertexai
# from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
# from google.cloud import aiplatform
# from IPython.display import display, HTML
# import pandas as pd
# from tqdm import tqdm
# from tqdm.asyncio import tqdm_asyncio
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# import aiohttp
# import time
# from collections import deque
# from concurrent.futures import ThreadPoolExecutor

# # Initialize Google Cloud project and location
# PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
# LOCATION = "us-central1"

# # Configuration constants
# MAX_CONCURRENT_REQUESTS = 50  # Adjust based on API limits
# RETRY_ATTEMPTS = 3
# BATCH_SIZE = 25
# REQUEST_TIMEOUT = 30  # seconds

# generation_config = {
#     "max_output_tokens": 8192,
#     "temperature": 1,
#     "top_p": 0.95,
# }

# safety_settings = [
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     ),
# ]

# class RequestRateLimiter:
#     def __init__(self, max_requests_per_second: int = 50):
#         self.max_requests_per_second = max_requests_per_second
#         self.request_times = deque(maxlen=max_requests_per_second)
#         self._lock = asyncio.Lock()

#     async def acquire(self):
#         async with self._lock:
#             current_time = time.time()
            
#             # Remove old timestamps
#             while self.request_times and current_time - self.request_times[0] > 1:
#                 self.request_times.popleft()
            
#             # If we've hit the limit, wait until enough time has passed
#             if len(self.request_times) >= self.max_requests_per_second:
#                 wait_time = 1 - (current_time - self.request_times[0])
#                 if wait_time > 0:
#                     await asyncio.sleep(wait_time)
            
#             self.request_times.append(time.time())

# class ModelPool:
#     def __init__(self, project_id: str, location: str, model_name: str, pool_size: int = 10):
#         self.project_id = project_id
#         self.location = location
#         self.model_name = model_name
#         self.pool_size = pool_size
#         self.models = asyncio.Queue(maxsize=pool_size)
#         self.initialized = False

#     async def initialize(self):
#         if not self.initialized:
#             vertexai.init(project=self.project_id, location=self.location)
#             for _ in range(self.pool_size):
#                 model = GenerativeModel(self.model_name)
#                 await self.models.put(model)
#             self.initialized = True

#     async def get_model(self):
#         if not self.initialized:
#             await self.initialize()
#         return await self.models.get()

#     async def release_model(self, model):
#         await self.models.put(model)

# @retry(
#     wait=wait_random_exponential(multiplier=1, max=10),
#     stop=stop_after_attempt(RETRY_ATTEMPTS)
# )
# async def process_single_prompt(
#     prompt: str,
#     model_pool: ModelPool,
#     rate_limiter: RequestRateLimiter,
#     index: int
# ) -> Tuple[int, str]:
#     """Process a single prompt with retry logic and return result with original index"""
#     await rate_limiter.acquire()
    
#     model = await model_pool.get_model()
#     try:
#         response = await model.generate_content_async(
#             [prompt],
#             generation_config=generation_config,
#             safety_settings=safety_settings,
#             stream=False,
#         )
#         return index, response.text
#     finally:
#         await model_pool.release_model(model)

# async def process_prompts_with_semaphore(
#     prompts: List[str],
#     model_pool: ModelPool,
#     rate_limiter: RequestRateLimiter,
#     semaphore: asyncio.Semaphore
# ) -> List[Tuple[int, str]]:
#     """Process prompts with concurrency control"""
#     async def process_with_semaphore(prompt: str, index: int) -> Tuple[int, str]:
#         async with semaphore:
#             return await process_single_prompt(prompt, model_pool, rate_limiter, index)

#     tasks = [
#         process_with_semaphore(prompt, i)
#         for i, prompt in enumerate(prompts)
#     ]
    
#     return await tqdm_asyncio.gather(*tasks, desc="Processing prompts")

# def get_input_text(active_ingredient_data, contraindication_text):
#     return f"Produce a list of diseases contraindicated for the active ingredient {active_ingredient_data} in the following contraindications list:\n{contraindication_text}Please format the list as ['item1', 'item2', ... ,'itemN']. Do not include any other text in the response. If no diseases are contraindicated for, return an empty list as '[]'. This code is being deployed in bulk so if the contraindications section is just <template> or similar, return an empty list."

# def generate_prompts(contraindications_data, active_ingredients_data, limit) -> List[str]:
#     print("Generating prompts...")
#     return [
#         get_input_text(active_ingredients_data[i], item)
#         for i, item in tqdm(
#             enumerate(contraindications_data[:limit]),
#             total=min(len(contraindications_data), limit)
#         )
#     ]

# async def main():
#     # Initialize pools and limiters
#     model_pool = ModelPool(
#         project_id=PROJECT_ID,
#         location=LOCATION,
#         model_name="gemini-1.5-pro-001",
#         pool_size=MAX_CONCURRENT_REQUESTS
#     )
#     rate_limiter = RequestRateLimiter(max_requests_per_second=50)
#     semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

#     # Load data
#     drugs_to_contraindications = pd.read_excel("../contraindicationList.xlsx")
#     contraindications_data = list(drugs_to_contraindications['contraindications'])
#     active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

#     # Generate prompts
#     prompts = generate_prompts(contraindications_data, active_ingredients_data, limit=1000)
#     print(f"Found {len(prompts)} prompts to process")

#     # Process prompts
#     results = await process_prompts_with_semaphore(
#         prompts,
#         model_pool,
#         rate_limiter,
#         semaphore
#     )

#     # Sort results by original index and extract responses
#     sorted_results = sorted(results, key=lambda x: x[0])
#     responses = [result[1] for result in sorted_results]
    
#     return active_ingredients_data, responses , contraindications_data # Both lists are identical and ordered

# if __name__ == "__main__":
#     active_ingredients_data, structured_lists, contraindications_text = asyncio.run(main())
#     data = pd.DataFrame({
#         "active ingredients": active_ingredients_data[0:len(structured_lists)],
#         "structured list": structured_lists,
#         "original text": contraindications_text[0:len(structured_lists)]
#     })
#     data.to_excel("indications_to_diseases.xlsx")



import asyncio
from typing import List, Tuple
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
from google.cloud import aiplatform
from IPython.display import display, HTML
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, wait_random_exponential
import concurrent.futures
from functools import partial

# Initialize Google Cloud project and location
PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
my_project = PROJECT_ID
LOCATION = "us-central1"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
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

@retry(wait=wait_random_exponential(multiplier=1, max=4))
async def process_single_prompt(prompt: str, project_id: str) -> str:
    """Process a single prompt with retry logic"""
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-001")
    
    response = await model.generate_content_async(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )
    return response.text

async def process_batch(batch_data: Tuple[List[str], int], project_id: str) -> Tuple[List[str], int]:
    """Process a batch of prompts and return results with batch index"""
    prompts, batch_idx = batch_data
    batch_results = []
    for prompt in prompts:
        result = await process_single_prompt(prompt, project_id)
        batch_results.append(result)
    return batch_results, batch_idx

def run_async_batch(batch_data: Tuple[List[str], int], project_id: str) -> Tuple[List[str], int]:
    """Wrapper to run async batch in a thread"""
    return asyncio.run(process_batch(batch_data, project_id))

async def process_all_prompts(prompts: List[str], project_id: str, batch_size: int = 10, max_workers: int = 4) -> List[str]:
    """Process all prompts in parallel batches while maintaining order"""
    # Split prompts into batches with indices
    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batches.append((batch, i // batch_size))

    # Process batches in parallel using ThreadPoolExecutor
    print(f"Processing {len(batches)} batches with {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with project_id
        process_func = partial(run_async_batch, project_id=project_id)
        # Process batches and get results with indices
        batch_results = list(tqdm(
            executor.map(process_func, batches), 
            total=len(batches),
            desc="Processing batches"
        ))

    # Sort results by batch index and flatten
    batch_results.sort(key=lambda x: x[1])  # Sort by batch index
    all_results = []
    for results, _ in batch_results:
        all_results.extend(results)
    
    return all_results

def get_input_text(active_ingredient_data, contraindication_text):
    text = "Produce a list of diseases contraindicated for the active ingredient " + str(active_ingredient_data) + " in the following contraindications list:\n" + str(contraindication_text) + "Please format the list as [\'item1\', \'item2\', ... ,\'itemN\']. Do not include any other text in the response. If no diseases are contraindicated for, return an empty list as \'[]\'. This code is being deployed in bulk so if the contraindications section is just \<template\> or similar, return an empty list."
    return text   

def generate_prompts(contraindications_data, active_ingredients_data, limit) -> List[str]:
    print("generating prompts...")
    prompts = []
    n_contraindications = len(contraindications_data)
    for index, item in tqdm(enumerate(contraindications_data), total=min(n_contraindications, limit)):
        if index < limit:
            prompts.append(get_input_text(active_ingredients_data[index], item))
    return prompts

async def main():
    # Load data
    drugs_to_contraindications = pd.read_excel("../contraindicationList.xlsx")
    contraindications_data = list(drugs_to_contraindications['contraindications'])
    active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

    # Generate prompts
    prompts = generate_prompts(contraindications_data, active_ingredients_data, limit=100)
    print(f"Found {len(prompts)} prompts to process")

    # Process all prompts in parallel batches
    responses = await process_all_prompts(
        prompts, 
        my_project,
        batch_size=10,  # Adjust batch size as needed
        max_workers=4   # Adjust number of parallel workers as needed
    )
    
    # Both lists are identical and ordered
    return responses, responses

if __name__ == "__main__":
    asyncio.run(main())