import pandas as pd
import tqdm
import tqdm.asyncio
import asyncio
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
import os
from tenacity import retry, wait_random_exponential
import time


# Set your Google Cloud project ID and location
PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
LOCATION = "us-central1"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@retry
async def predict_gemini_model(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    content: str,
) -> str:
    """Predict using a Gemini Model with safety settings."""
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
    return "NO LLM OUTPUT"  # Return empty string if no text content found

async def run_multiple_predictions(prompts, model_name="gemini-1.5-flash-001"):
    tasks = []
    for prompt in tqdm.tqdm(prompts, total=len(prompts)):
        task = predict_gemini_model(
            model_name=model_name,
            temperature=0.2,
            max_output_tokens=256,
            top_p=0.8,
            top_k=40,
            content=prompt,
        )
        tasks.append(task)
    
    responses = []
    for f in tqdm.asyncio.tqdm.as_completed(tasks):
        responses.append(await f)

    #responses = await asyncio.gather(*tasks)
    return responses

def get_input_text(active_ingredient_data, contraindication_text):
    text = "Produce a list of diseases contraindicated for the active ingredient " + str(active_ingredient_data) + " based on the following contraindications list:\n" + str(contraindication_text) + "Please format the list as [\'item1\', \'item2\', ... ,\'itemN\']. Do not include any other text in the response. If no diseases are contraindicated for, return an empty list as \'[]\'. If the drug is only used for diagnostic purposes, return \'diagnostic/contrast/radiolabel\'. Do not include hypersensitivity or allergy to the named drug as a contraindication. This code is being deployed in bulk so if the contraindications section is just \'template\' or similar, return an empty list. Only include conditions that the drug causes, and omit pre-exisitng conditions of the patients."
    return text   

def generate_prompts(contraindications_data, active_ingredients_data, limit) -> list[str]:
    print("generating prompts...")
    prompts = []
    n_contraindications = len(contraindications_data)
    for index, item in tqdm.tqdm(enumerate(contraindications_data), total=n_contraindications):
        if index < limit:
            prompts.append(get_input_text(active_ingredients_data[index], item))
    return prompts


async def main():
    
    drugs_to_contraindications = pd.read_excel("../contraindicationList.xlsx")
    contraindications_data = list(drugs_to_contraindications['contraindications'])
    active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

    limit = len(contraindications_data)
    #limit = 200
    prompts = generate_prompts(contraindications_data, active_ingredients_data, limit=limit)
    print("found ", len(prompts), " prompts to feed to LLM API")
    start_time = time.time()
    responses = await run_multiple_predictions(prompts)
    for response in responses:
        print(f"Response: {response}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(prompts)} prompts in {elapsed_time} seconds")

    

    data = pd.DataFrame({
        "Active Ingredients": active_ingredients_data[0:limit],
        "Structured Disease list": responses,
        "Source Text": contraindications_data[0:limit]
    })

    data.to_excel("active_ingredients_to_structured_lists_v2.xlsx")


if __name__ == "__main__":
    asyncio.run(main())