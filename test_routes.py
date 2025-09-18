import asyncio

import requests

from commons.cache.redis import RedisCache

BASE_URL = "http://127.0.0.1:5003/api"
r = RedisCache()


def test_get_question():
    """Calls the "/generate-question endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/generate-question")
        response.raise_for_status()  # Raise an exception for bad status codes
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the question: {e}")
        return None, None


def test_get_answer(qa_id: str):
    """calls "/generate-answer" endpoint"""
    if not qa_id:
        print("No qa_id provided.")
        return None

    print(f"\nRequesting answer for QA ID: {qa_id}")
    data = {"qa_id": qa_id}
    try:
        response = requests.post(f"{BASE_URL}/generate-answer", json=data)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and data.get("ans_id"):
            print(f"Received answer: {data['ans_id']}")
            return data["ans_id"]
        else:
            error_message = data.get("error", "The API did not return a valid answer.")
            print(f"Failed to get answer: {error_message}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the answer: {e}")
        return None


def test_get_question_augment(base_question: str, num_augments: int):
    """calls "/get-question-augment" endpoint"""
    if not base_question:
        print("No base_question provided.")
        return None

    print(f"\nRequesting augments for question: '{base_question}'...")
    data = {"question": base_question, "num_augments": num_augments}
    try:
        response = requests.post(f"{BASE_URL}/get-question-augment", json=data)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and data.get("augments"):
            print("Received augments:")
            for augment in data["augments"]:
                print(f"  - {augment}")
            return data["augments"]
        else:
            error_message = data.get("error", "The API did not return valid augments.")
            print(f"Failed to get augments: {error_message}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the augments: {e}")
        return None


def test_order_answer(question: str):
    """Calls the order-answer endpoint."""
    if not question:
        print("No question provided.")
        return None

    print(f" order_answer for question: '{question}' \n")
    data = {"question": question}
    try:
        response = requests.post(f"{BASE_URL}/order-answer", json=data)
        response.raise_for_status()
        data = response.json()
        print(f"Order answer response: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the answer: {e}")
        return None


def test_pop_qa(qa_id: str):
    """Calls the pop-qa endpoint."""
    if not qa_id:
        print("No qa_id provided.")
        return None

    print(f" pop_qa for qa_id: '{qa_id}' \n")
    data = {"qa_id": qa_id}
    try:
        response = requests.post(f"{BASE_URL}/pop-qa", json=data)
        response.raise_for_status()
        data = response.json()
        print(f"Pop QA response: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while popping the QA pair: {e}")
        return None


async def main():
    """
    - this is a quick hacky way to test the endpoints from synthetic_gen.py
    - comment / uncomment out blocks to test different endpoints.
    - python test_routes.py to run.
    """

    # test getting question
    question = test_get_question()
    print(f"Question: {question.json()}")  # type: ignore

    # test getting answer with qa_id
    qa_id = question.json()["qa_id"]  # type: ignore
    answer = test_get_answer(qa_id)
    print(f"Answer: {answer}")
    qa_id = "7fd71725-9d47-4bf8-9801-7907357f006a"
    # # # test getting question augment
    # base_question = question.json()["prompt"]  # type: ignore
    # question_augment = test_get_question_augment(base_question, num_augments=2)  # type: ignore
    # print(f"Question Augment: {question_augment}")

    # retrieve augmented qn from redis once its finished generating.
    # uncomment this block once youve confirmed the augment_id exists in redis.
    # if question_augment:
    #     augment_id = (
    #         "61639d33-1808-43ce-b27e-24de657e0b0a"  # replace w id that youve generated.
    #     )
    #     key = r._build_key(r._qn_augment_key, augment_id)
    #     augmented_qn_bytes = await r.redis.get(key)
    #     if augmented_qn_bytes:
    #         augmented_qn = json.loads(augmented_qn_bytes)
    #         print(f"  - Augmented Question ({augment_id}): {augmented_qn}")
    #     else:
    #         print(f"  - Could not retrieve augmented question for id {augment_id}")

    # # test ordering answer
    # # order_qn = """
    # # Write a program that implements tic-tac-toe with a 3x3 board using javascript and html.
    # # """
    # order_qn_2 = question.json()["prompt"]  # type: ignore
    # ordered_answer_id = test_order_answer(question=order_qn_2)
    # print(f"Ordered Answer ID: {ordered_answer_id}")

    # # test getting answer with qa_id
    # uncomment this block once youve confirmed the ordered_answer_id finished generating.
    # target_id = ordered_answer_id["ans_id"] #type ignore
    # target_id = "d1cffb22-132f-4927-9d82-a00cec5b7897"
    # answer = test_get_answer(target_id)  # type: ignore
    # print(f"Answer: {answer}")

    # test popping qa
    test_pop_qa(qa_id)


if __name__ == "__main__":
    asyncio.run(main())
