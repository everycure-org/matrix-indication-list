import pandas as pd
import asyncio
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from typing import List, Any
from dataclasses import dataclass
import time
from tqdm import tqdm
import math

# Set your Google Cloud project ID and location
PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
LOCATION = "us-central1"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

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
                    max_output_tokens=256,
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
    with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
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

def generate_prompts(contraindications_data, active_ingredients_data, limit) -> List[str]:
    print("Generating prompts...")
    prompts = []
    n_contraindications = min(len(contraindications_data), limit)
    
    for index in range(n_contraindications):
        prompts.append(get_input_text(active_ingredients_data[index], contraindications_data[index]))
    
    return prompts

async def main():
    # Configuration
    BATCH_SIZE = 50  # Number of prompts to process in each batch
    MAX_CONCURRENT = 10  # Maximum number of concurrent API calls
    REQUESTS_PER_MINUTE = 200  # Maximum number of API requests per minute
    
    # Load data
    drugs_to_contraindications = pd.read_excel("../contraindicationList.xlsx")
    contraindications_data = list(drugs_to_contraindications['contraindications'])
    active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

    limit = len(contraindications_data)
    prompts = generate_prompts(contraindications_data, active_ingredients_data, limit=limit)
    
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
    data = pd.DataFrame({
        "Active Ingredients": active_ingredients_data[0:limit],
        "Structured Disease list": responses,
        "Source Text": contraindications_data[0:limit]
    })

    data.to_excel("active_ingredients_to_structured_lists_v2.xlsx")

if __name__ == "__main__":
    asyncio.run(main())