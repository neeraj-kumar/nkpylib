"""Async code examples."""

import requests
import asyncio
import concurrent.futures

# Define a synchronous function to make a POST request using the requests library
def post_request(url, data):
    response = requests.post(url, json=data)
    return response.json()

# Define an asynchronous function to run the synchronous POST request in a separate thread
async def post_request_async(url, data):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, post_request, url, data)
    return response

# Example URL and data
url = "https://jsonplaceholder.typicode.com/posts"
data = {"title": "foo", "body": "bar", "userId": 1}

# Synchronous POST request
sync_response = post_request(url, data)
print("Synchronous response:", sync_response)

# Asynchronous POST request
async def main():
    async_response = await post_request_async(url, data)
    print("Asynchronous response:", async_response)

# Run the async main function
asyncio.run(main())

