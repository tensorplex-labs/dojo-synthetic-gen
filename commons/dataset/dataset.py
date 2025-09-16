"""
use to manually generate sample datasets for testing.
"""

import asyncio
import json

from commons.cache import RedisCache


async def get_all_redis_list_elements():
    r = RedisCache()
    all_elements = []
    current_qa = 0
    # Keep dequeuing until we get None (empty list)
    while True:
        element = await r.dequeue()
        if element is None:
            break
        current_qa += 1
        print(f"Processing QA pair {current_qa}")
        try:
            # Parse the JSON string into a Python object
            parsed_element = json.loads(element)
            all_elements.append(parsed_element)
        except json.JSONDecodeError:
            # Skip invalid JSON elements
            continue

    with open("output.json", "w") as f:
        # Write the list directly to the file
        json.dump(all_elements, f, indent=2)


if __name__ == "__main__":
    asyncio.run(get_all_redis_list_elements())
