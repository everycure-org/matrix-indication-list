import aiohttp
import asyncio

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return responses

if __name__ == "__main__":
    urls = [
        "https://api.github.com",
        "https://api.github.com/users/github",
        "https://api.github.com/repos/github/github"
    ]
    
    responses = asyncio.run(main(urls))
    
    # Populate responses into a vector (list in Python)
    response_vector = []
    for response in responses:
        response_vector.append(response)
    
    print(f"Collected {len(response_vector)} responses")
    print(f"First response: {response_vector[0][:100]}...")  # Print first 100 characters of the first response