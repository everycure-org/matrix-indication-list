import asyncio
from typing import List
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
from google.cloud import aiplatform
from IPython.display import display, HTML
import pandas as pd
from tqdm import tqdm
from tenacity import retry, wait_random_exponential

# Initialize Google Cloud project and location
PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
my_project = PROJECT_ID
LOCATION = "us-central1"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Initialize the Gemini model
model = GenerativeModel("gemini-1.5-flash-001")


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

async def generate_content(prompt: str) -> str:
    response = await model.generate_content_async(prompt)
    return response.text

async def process_responses(responses: List[asyncio.Task]) -> List[str]:
    results = []
    for response in responses:
        result = await response
        results.append(result)
    return results

async def generate_responses(prompts: List[str]) -> List[str]:
    tasks = [asyncio.create_task(generate_content(prompt)) for prompt in prompts]
    responses = await process_responses(tasks)
    return responses

def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

def display_responses(prompts: List[str], responses: List[str]):
    html = "<table><tr><th>Prompt</th><th>Response</th></tr>"
    for prompt, response in zip(prompts, responses):
        html += f"<tr><td>{prompt}</td><td>{response}</td></tr>"
    html += "</table>"
    display(HTML(html))

def get_input_text(active_ingredient_data, contraindication_text):
    text = f"Produce a list of diseases contraindicated for the active ingredient {active_ingredient_data} in the following contraindications list:\n {contraindication_text} Please format the list as [\'item1\', \'item2\', ... ,\'itemN\']. Do not include any other text in the response. If no diseases are contraindicated for, return an empty list as \'[]\'. If the drug is only used for diagnostic purposes, return \'diagnostic/contrast/radiolabel\'. Do not include hypersensitivity or allergy to the named drug as a contraindication. This code is being deployed in bulk so if the contraindications section is just \<template\> or similar, return an empty list."
    return text   

def generate_prompts(contraindications_data, active_ingredients_data, limit) -> list[str]:
    print("generating prompts...")
    prompts = []
    n_contraindications = len(contraindications_data)
    for index, item in tqdm(enumerate(contraindications_data), total=n_contraindications):
        if index < limit:
            prompts.append(get_input_text(active_ingredients_data[index], item))
    return prompts

@retry(wait=wait_random_exponential(multiplier=1, max=120))
async def async_generate(prompt, my_project):
  vertexai.init(project=my_project, location="us-central1")
  model = GenerativeModel(
    "gemini-1.5-pro-001",
  )
  response = await model.generate_content_async(
      [prompt],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=False,
  )

  return response.text

async def main():
    drugs_to_contraindications = pd.read_excel("../contraindicationList.xlsx")
    contraindications_data = list(drugs_to_contraindications['contraindications'])
    active_ingredients_data = list(drugs_to_contraindications['active ingredient'])

    prompts = generate_prompts(contraindications_data, active_ingredients_data, limit=100)
    print("found ", len(prompts), " prompts to feed to LLM API")

    get_responses = [async_generate(prompt, my_project) for prompt in prompts]

    responses = await run_async(generate_responses(prompts))
    structuredLists = await asyncio.gather(*get_responses)

    #display_responses(prompts, responses)

if __name__ == "__main__":
    main()