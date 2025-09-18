"""
main logic for generating js/html question-answer pairs.
"""

import json
import os
import random
import uuid
from asyncio import create_task, sleep
from enum import Enum
from typing import cast

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from instructor import AsyncInstructor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# from langfuse.decorators import get_client, observe
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from loguru import logger
from openai import AuthenticationError, PermissionDeniedError
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
)

from commons.augmenter import Augmenter
from commons.augmenter.types import QuestionAugmentation
from commons.cache.redis import RedisCache
from commons.config import ANSWER_MODELS, GENERATOR_MODELS
from commons.dataset.personas import get_random_persona
from commons.linter import lint_and_fix_code
from commons.llm import Provider, _get_llm_api_kwargs, call_llm, get_llm_api_client
from commons.prompt_builders import (
    additional_notes_for_question_prompt,
    build_code_answer_prompt,
    build_code_generation_question_prompt,
)
from commons.types import (
    CodeAnswer,
    CodeQuestion,
    GeneratedAnswer,
    Topics,
)
from commons.utils.logging import log_to_langfuse

load_dotenv()

r = RedisCache()


class AugmentStrategy(Enum):
    CHANGE_QUESTIONS = 0
    CHANGE_ANSWERS = 1
    PERFORMANCE = 2


@observe(as_type="generation", capture_input=True, capture_output=True)
async def generate_question(
    client: AsyncInstructor,
    model: str,
    _topic: Topics,
    persona: str,
):
    try:
        kwargs = {
            "response_model": CodeQuestion,
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": build_code_generation_question_prompt(
                        random.choices([2, 3], weights=[0.5, 0.5])[0],
                        topic=_topic,
                        persona=persona,
                    ),
                }
            ],
            "temperature": random.uniform(0, 1),
            "max_tokens": 64000,
            "max_retries": AsyncRetrying(stop=stop_after_attempt(1), reraise=True),
            "top_p": random.uniform(0, 0.8),
            "seed": random.randint(0, int(1e9)),  # needed for OpenAI
        }
        response_model, completion = await call_llm(client, kwargs)
        coding_question = response_model.question

        # retry if generated question is under 100 tokens
        # @dev WARNING: if question generation persistently fails (eg. model provider is down), syn-API will spam attempts to generate questions.
        if completion.usage.completion_tokens < 100:
            raise Exception("Incomplete generation, question is under 100 tokens")
        coding_question = additional_notes_for_question_prompt(coding_question)

        kwargs["topic"] = _topic.name
        log_to_langfuse(kwargs, response_model, completion)
        logger.info("Base Question Generation Completed")
        return coding_question
    except Exception as e:
        logger.error(f"Error occurred while generating question: {e}")
        raise


@observe(as_type="generation", capture_input=True, capture_output=True)
async def generate_answer(
    client: AsyncInstructor,
    model: str,
    question: str,
    topic: Topics,
    augment: int | None = None,
) -> GeneratedAnswer:
    """Generates a coding question answer for a given coding question."""

    ans_id = str(uuid.uuid4())
    # this is a hack because CodeAnswer.model_json_schema cannot be imported by prompt_builders without a ciruclar import error.add()
    # need to move where these types are declared during refactor.
    _answer_format = CodeAnswer.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": build_code_answer_prompt(
                question,
                False,  # @dev turned off for v2
                topic=topic,
                answer_format=_answer_format,
            ),
        },
    ]

    kwargs = {
        "response_model": CodeAnswer,
        "model": model,
        "messages": messages,
        "max_retries": AsyncRetrying(stop=stop_after_attempt(1), reraise=True),
        "temperature": random.uniform(0, 0.1),
        "max_tokens": 64000,
        "top_p": random.uniform(0, 0.6),
        "response_format": {
            "type": "json_object",
            "response_schema": CodeAnswer.model_json_schema(),
            "enforce_validation": True,
        },
    }
    if model.startswith("openai"):
        kwargs["seed"] = random.randint(0, cast(int, 1e9))  # needed for OpenAI

    try:
        response_model, completion = await call_llm(client, kwargs)
        kwargs["question"] = question
        kwargs["ans_id"] = ans_id
        log_to_langfuse(kwargs, response_model, completion)
        logger.info(f"{ans_id} Answer Generation Completed ")

        # execute auto-linting and use LLm to fix syntax errors if any. Will modify the response_model in place.
        response_model = await lint_and_fix_code(client, model, response_model, ans_id)

        return GeneratedAnswer(
            model=model, answer=response_model, id=ans_id, augment=augment
        )
    except Exception as e:
        logger.error(f"Error while generating {ans_id} answer: {e}")
        raise


def _build_single_index_html(ans: CodeAnswer) -> CodeAnswer:
    file_extensions = set(os.path.splitext(file.filename)[1] for file in ans.files)
    logger.trace(f"found file extensions from CodeAnswer: {file_extensions}")
    has_js = ".js" in file_extensions
    has_css = ".css" in file_extensions
    has_html = ".html" in file_extensions

    if not has_html:
        raise ValueError("No HTML file found in the CodeAnswer")

    index_html_file = [
        f for f in ans.files if os.path.splitext(f.filename)[1] == ".html"
    ]
    assert len(index_html_file) == 1
    index_html = index_html_file[0]
    soup = BeautifulSoup(index_html.content, "html.parser")

    # Ensure we have html and head tags
    html_tag = soup.find("html")
    if not html_tag:
        html_tag = soup.new_tag("html")
        soup.append(html_tag)

    head_tag = soup.find("head")
    if not head_tag:
        head_tag = soup.new_tag("head")
        html_tag.insert(0, head_tag)

    body_tag = soup.find("body")
    if not body_tag:
        body_tag = soup.new_tag("body")
        html_tag.append(body_tag)

    if has_css:
        index_css_file = [
            f for f in ans.files if os.path.splitext(f.filename)[1] == ".css"
        ]
        assert len(index_css_file) == 1
        index_css = index_css_file[0]
        style_tag = soup.new_tag("style")
        style_tag.string = index_css.content
        head_tag.append(style_tag)

    if has_js:
        index_js_file = [
            f for f in ans.files if os.path.splitext(f.filename)[1] == ".js"
        ]
        assert len(index_js_file) == 1
        index_js = index_js_file[0]
        script_tag = soup.new_tag("script")
        script_tag.string = index_js.content
        body_tag.append(script_tag)

    # Keep only the HTML file, removing JS and CSS files
    new_files = [
        f for f in ans.files if os.path.splitext(f.filename)[1] not in [".js", ".css"]
    ]
    # Update the content of the HTML file
    for file in new_files:
        if file.filename == "index.html":
            file.content = str(soup)

    return CodeAnswer(files=new_files)


# merges output index.js into index.html
def merge_js_and_html(result: CodeAnswer) -> CodeAnswer:
    ans_with_index_html = _build_single_index_html(result)
    html_file = next(
        (file for file in ans_with_index_html.files if file.filename == "index.html"),
        None,
    )
    if html_file:
        pass
    else:
        raise ValueError("No index.html file found in the answer")

    # replace whole CodeAnswer with a single final_html file
    result.files = [file for file in result.files if file.filename == "index.html"]

    # replace old html with updated html with inlined JS and CSS.
    if result.files:
        # result.files[0].content = final_html
        result.files[0].content = html_file.content
    else:
        raise ValueError("No index.html file found in the result")
    return result


# use trace to avoid double dipping cost logging on nested observations
@observe(as_type="generation")
async def build_prompt_responses_pair():
    client = get_llm_api_client()
    question_model = random.choice(GENERATOR_MODELS)
    answer_model = random.choice(ANSWER_MODELS)
    augmenter = Augmenter(client, answer_model)
    final_answers: list[GeneratedAnswer] = []

    # 1. get random persona from hugging face
    persona = get_random_persona()

    # 2. randomly select a topic. change weights accordingly to choose what topic of Tasks to generate.
    selected_topic = random.choices(list(Topics), weights=[0.4, 0.3, 0.3], k=1)[0]
    try:
        # 3. generate base question using the topic
        question_prompt = await generate_question(
            client, question_model, selected_topic, persona
        )
        if question_prompt is None:
            raise ValueError("generate_question() returned null")

        # 4. generate base answer
        base_answer = await generate_answer(
            client, answer_model, question_prompt, selected_topic, 0
        )
        if base_answer is None:
            raise ValueError("generate_answer() returned null")

        # 5. generate unified augments
        final_answers = [base_answer] + await augmenter.run_unified_augment(
            question_prompt, base_answer.answer, selected_topic
        )

    except (AuthenticationError, PermissionDeniedError) as e:
        logger.error(f"Fatal Error when generating question-answer pair: {e}")
        raise e
    except Exception as e:
        raise e

    # parse QA pairs, format and return response.
    responses = []
    synthetic_ground_truth: dict[str, int] = {}
    for ans in final_answers:
        if not ans:
            raise RuntimeError("Error generating prompt-response pair")

        # merge generated JS code into HTML file
        result = merge_js_and_html(ans.answer)
        formatted_files = [
            {"filename": file.filename, "content": file.content}
            for file in result.files
        ]

        responses.append(
            {
                "model": ans.model,
                "completion": {"files": formatted_files},
                "cid": ans.id,
            }
        )

        if ans.augment is not None:
            logger.debug(f"{ans.model=},{ans.id=}, {ans.augment=}")
            synthetic_ground_truth[ans.id] = ans.augment

    # this is the return payload from the task creation API route
    # @dev the commented fields are not injested by dojo.
    return {
        "prompt": question_prompt,
        # "question_model": question_model,
        "responses": responses,
        "ground_truth": synthetic_ground_truth,
        # "augmented_prompts": augmented_prompts,
        # "topic": selected_topic.name,
        # "persona": persona,
        "metadata": {
            "augment_type": "UNIFIED",
        },
    }


@observe(as_type="generation")
async def run_dojo_v2_process():
    """
    - runs as a background task, executed by Worker.py.
    - generates questions and stores them in question redis queue.
    - then generates answers from the questions and stores them in an answer redis queue
    - also will generate augmented answers and store them in redis.
    - dojo can later query answer queue with the uid returned from question queue.

    @returns answer_payload
    """
    client = get_llm_api_client()
    question_model = random.choice(GENERATOR_MODELS)
    answer_model = random.choice(ANSWER_MODELS)
    persona = get_random_persona()
    selected_topic = random.choices(list(Topics), weights=[0.4, 0.3, 0.3], k=1)[0]
    try:
        # generate base question
        question_prompt = await generate_question(
            client, question_model, selected_topic, persona
        )

        # generate base answer
        ans = await generate_answer(
            client, answer_model, question_prompt, selected_topic
        )
        # ans = await generate_answer_lchain(
        #     answer_model, question_prompt, selected_topic
        # )
        ans_payload = _make_answer_payload(ans, question_prompt)

        # generate 1 random negatively augmented answer
        augmenter = Augmenter(client, random.choice(GENERATOR_MODELS))
        augment_types = [
            a for a in QuestionAugmentation if a != QuestionAugmentation.ORIGINAL
        ]
        selected_augment = random.choice(augment_types)
        augmented_question = await augmenter._augment_question(
            question_prompt, selected_augment
        )
        # aug_ans = await generate_answer_lchain(
        #     answer_model, augmented_question.question, selected_topic
        # )
        aug_ans = await generate_answer(
            client, answer_model, augmented_question.question, selected_topic
        )
        aug_ans_payload = _make_answer_payload(aug_ans, augmented_question.question)

    except Exception as e:
        logger.error(f"Error running dojo v2 process: {e}")
        raise e

    # once all components are generated, store them in redis
    try:
        qa_id = ans.id
        await r.store_answer(aug_ans_payload)
        await r.store_question(qa_id, question_prompt, aug_ans.id)
    except Exception as e:
        logger.error(f"Error storing answer and question in redis: {e}")
        raise e
    # @to-do create a type for the return payload.
    # @dev whatever is returned will be store to redis by worker.py
    return ans_payload


@observe(as_type="generation")
async def v2_get_augment_questions(base_question: str, num_augments: int):
    """
    - driver function that executes generation of num_augments augmented questions
    @param base_question: the base question to augment
    @return augment_ids: list of uids for each augmented question that can be used to retrieve the augmented question from redis.
    """
    client = get_llm_api_client()
    augmenter = Augmenter(client, random.choice(GENERATOR_MODELS))
    try:
        augmented_questions = await augmenter.v2_run_augment_question(
            base_question, num_augments
        )

        augment_ids = []
        for qn in augmented_questions:
            augment_id = str(uuid.uuid4())
            await r.store_augmented_question(qn.question, augment_id)
            augment_ids.append(augment_id)
    except Exception as e:
        logger.error(f"Error v2_augment_question: {e}")
        raise e
    return augment_ids


@observe(as_type="generation", capture_input=True, capture_output=True)
async def v2_run_order_answer(question: str):
    """
    - function to bespokely generate code answers with any prompt.
    - intended to be used to generate with augmented prompts.
    - generates and stores answer in redis
    @ return answer_id that can be used to retrieve the answer from redis.
    """
    await sleep(0)  # @dev stupid hack to silence ruff hook
    answer_id = str(uuid.uuid4())
    create_task(v2_cook_order(answer_id, question))
    return answer_id


async def v2_cook_order(ans_id: str, question: str):
    client = get_llm_api_client()
    answer_model = random.choice(ANSWER_MODELS)
    ans = await generate_answer(client, answer_model, question, Topics.ANIMATION)
    ans_payload = _make_answer_payload(ans, question)
    ans_payload["qa_id"] = ans_id
    await r.store_answer(ans_payload)


def _make_answer_payload(ans: GeneratedAnswer, question: str) -> dict:
    merged_answer = merge_js_and_html(ans.answer)
    formatted_files = [
        {"filename": file.filename, "content": file.content}
        for file in merged_answer.files
    ]
    responses = [
        {
            "model": ans.model,
            "completion": {"files": formatted_files},
            "cid": ans.id,
        }
    ]
    return {
        "prompt": question,
        "responses": responses,
        "qa_id": ans.id,
    }


@observe(as_type="generation", capture_input=True, capture_output=True)
async def generate_answer_lchain(
    model: str,
    question: str,
    topic: Topics,
    augment: int | None = None,
) -> GeneratedAnswer:
    """
    replicates the functionality of generate_answer from synthetic.py using langchain
    """
    ans_id = str(uuid.uuid4())
    langfuse_handler = CallbackHandler()
    langfuse = get_client()
    _answer_format = CodeAnswer.model_json_schema()

    answer_format_str = json.dumps(_answer_format).replace("{", "{{").replace("}", "}}")

    prompt_str = build_code_answer_prompt(
        question,
        False,  # @dev turned off for v2
        topic=topic,
        answer_format=answer_format_str,
    )

    prompt = ChatPromptTemplate.from_messages([("system", prompt_str)])

    creds = _get_llm_api_kwargs(Provider.OPENROUTER)

    llm_params = {
        "temperature": random.uniform(0, 0.1),
        "top_p": random.uniform(0, 0.6),
        "seed": random.randint(0, cast(int, 1e9)),
        "max_tokens": 64000,
    }

    llm = ChatOpenAI(
        api_key=SecretStr(creds["api_key"]),
        base_url=creds["base_url"],
        model=model,
        temperature=llm_params["temperature"],
        top_p=llm_params["top_p"],
        max_retries=0,
        seed=llm_params["seed"],
    ).with_structured_output(CodeAnswer, include_raw=True)

    chain = prompt | llm

    response_dict = await chain.ainvoke({}, config={"callbacks": [langfuse_handler]})
    response_model = response_dict["parsed"]
    if response_model is None:
        raise ValueError("Failed to parse LLM response into CodeAnswer")

    # for lint_and_fix_code
    client = get_llm_api_client()
    response_model = await lint_and_fix_code(client, model, response_model, ans_id)

    logger.info(f"{ans_id} Answer Generation Completed ")
    usage = response_dict["raw"].usage_metadata
    langfuse.update_current_generation(
        input=question,
        output=response_model.model_dump(),
        usage_details={
            "input": usage["input_tokens"],
            "output": usage["output_tokens"],
        },
    )
    return GeneratedAnswer(
        model=model, answer=response_model, id=ans_id, augment=augment
    )
