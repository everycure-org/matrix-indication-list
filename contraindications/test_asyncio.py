import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch(session, name, biolink_class):
    url = f'https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=false&offset=0&limit=10&biolink_type={biolink_class}'
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()

async def fetch_with_progress(session, name, biolink_class, pbar):
    try:
        result = await fetch(session, name, biolink_class)
    except Exception as e:
        result = e
    pbar.update(1)
    return result

async def main(names, biolink_class):
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(names), desc="Processing requests")
        tasks = [fetch_with_progress(session, name, biolink_class, pbar) for name in names]
        responses = await asyncio.gather(*tasks)
        pbar.close()
    return responses

# Example usage
names = ["aspirin", "ibuprofen", "acetaminophen"]
biolink_class = "ChemicalSubstance"

# Run the async function
responses = asyncio.run(main(names, biolink_class))

# Print responses (they will be in the same order as the input)
for name, response in zip(names, responses):
    print(f"Response for {name}:")
    if isinstance(response, Exception):
        print(f"Error: {response}")
    else:
        print(response)
    print()