import asyncio
from typing import List
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import aiplatform
from IPython.display import display, HTML

# Initialize Google Cloud project and location
PROJECT_ID = "mtrx-wg2-modeling-dev-9yj"
LOCATION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Initialize the Gemini model
model = GenerativeModel("gemini-1.5-flash-001")

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

# Example usage in a Jupyter notebook cell:
prompts = [
    "What is the capital of France?",
    "Explain the concept of machine learning in simple terms.",
    "Write a haiku about programming.",
]

responses = run_async(generate_responses(prompts))
display_responses(prompts, responses)