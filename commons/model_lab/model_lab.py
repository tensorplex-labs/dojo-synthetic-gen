import asyncio
import os
import random

from dotenv import load_dotenv
from instructor import AsyncInstructor

# from langfuse.decorators import observe
from langfuse import observe
from loguru import logger

from commons.dataset.personas import get_random_persona, load_persona_dataset
from commons.llm import call_llm, get_llm_api_client
from commons.synthetic import (
    CodeAnswer,
    generate_answer,
    generate_question,
    merge_js_and_html,
)
from commons.types import Topics
from commons.utils.logging import log_to_langfuse

load_dotenv()

"""
    model_lab.py
    1. generate a question from each topic
    2. send each question to each model
    3. save all results as a json file

    instructions
    - edit the question_model and answer_models variables with the desired models
    - question_model will be used to generate the question
    - each answer model will generate code for that question.
    - to run the script: python -m commons.model_lab.model_lab
    - output will be saved to whatever is defined in the OUTPUT_FILE variable
"""

# get model names from openrouter website

question_model = "anthropic/claude-3.5-sonnet"
question_model = "google/gemini-2.5-flash-preview"

answer_models = [
    # "deepseek/deepseek-r1",
    # "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-flash-preview",
    # "deepseek/deepseek-chat-v3-0324:free",
    # "qwen/qwen2.5-32b-instruct",  # 0.79/M
    # "qwen/qwq-32b",  # 0.12/M in 0.18/M out
    # # "anthropic/claude-3.7-sonnet",
    # # "anthropic/claude-3.5-sonnet",
    # "anthropic/claude-3.5-haiku",
    # "anthropic/claude-3.5-haiku:beta",
]
OUTPUT_FILE = "syn-gen-trials.json"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


async def main():
    logger.info("Starting standalone synthetic data generation")

    load_persona_dataset()

    client = get_llm_api_client()

    try:
        # 1. generate questions from all topics concurrently
        question_tasks = []
        for topic in Topics:
            logger.info(f"generating {topic.name} question ...")
            persona = get_random_persona()
            task = generate_question(client, question_model, topic, persona)
            question_tasks.append(task)

        # Wait for all questions to be generated
        question_results = await asyncio.gather(*question_tasks)
        questions = [
            {"topic": topic, "question": question}
            for topic, question in zip(Topics, question_results, strict=False)
        ]

        # 2. generate answers for all questions concurrently
        answer_tasks = []
        for model in answer_models:
            for q in questions:
                logger.info(
                    f"generating {q['topic'].name} answer with model: {model} ..."
                )
                task = generate_answer(client, model, q["question"], q["topic"])
                answer_tasks.append((model, q, task))

        # Wait for all answers to be generated
        answers = []
        answer_results = await asyncio.gather(
            *[task for _, _, task in answer_tasks], return_exceptions=True
        )

        # Process results, handling any exceptions
        for (model, q, _), result in zip(answer_tasks, answer_results, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    f"Error generating {q['topic'].name} answer with model: {model}: {result}"
                )
                continue

            # Only process successful results (result is GeneratedAnswer)
            from commons.types import GeneratedAnswer

            if not isinstance(result, GeneratedAnswer):
                logger.error(
                    f"Unexpected result type for {q['topic'].name} answer with model: {model}"
                )
                continue

            logger.success(f"Generated {q['topic'].name} answer with model: {model}")
            # merge generated index.js into index.html
            merged_answer = merge_js_and_html(result.answer)
            answers.append(
                {
                    "name": f"{model}_{q['topic'].name}",
                    "answer": merged_answer,
                    "question": q["question"],
                }
            )

        # 3. save answers to file
        result = []
        for ans in answers:
            # Convert each CodeAnswer to dict and add to result list
            result.append(
                {
                    "files": [
                        {
                            "tag": ans["name"],
                            "filename": file.filename,
                            "content": file.content,
                            "language": file.language,
                            "question": ans["question"],
                        }
                        for file in ans["answer"].files
                    ]
                }
            )

        # Save to file in the current directory
        output_path = os.path.join(CURRENT_DIR, OUTPUT_FILE)
        import json

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error running model_lab.py: {e}")
        raise


@observe(as_type="generation", capture_input=True, capture_output=True)
async def gen_reasoning(model: str, prompt: str, persona: str, client: AsyncInstructor):
    logger.info(f"Generating with model: {model} for {persona}")
    kwargs = {
        "response_model": CodeAnswer,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64000,
        "top_p": random.uniform(0.9, 1.0),
        "response_format": {
            "type": "json_object",
            "response_schema": CodeAnswer.model_json_schema(),
            "enforce_validation": True,
        },
    }

    res, completion = await call_llm(client, kwargs)

    # Get model name for file naming
    if model == "google/gemini-2.5-flash-preview:thinking":
        model_name = "flash-thinking"
    else:
        model_name = kwargs["model"].split("/")[-1]  # Extract last part of model path

    # log to langfuse
    kwargs["persona"] = persona
    log_to_langfuse(kwargs, res, completion)

    # Save individual files
    for file in res.files:
        # Determine file extension and create filename
        filename = f"{model_name}{os.path.splitext(file.filename)[1]}"
        output_path = os.path.join(CURRENT_DIR, filename)

        # Write content to file
        with open(output_path, "w") as f:
            f.write(file.content)
        logger.info(f"Saved {filename} to {output_path}")


async def run_reasoning_generation():
    load_persona_dataset()

    logger.info("Starting reasoning model generation")
    reasoning_models = [
        # "deepseek/deepseek-r1",
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-flash-preview",
        "google/gemini-2.5-flash-preview:thinking",
        # "anthropic/claude-3.7-sonnet",
    ]
    client = get_llm_api_client()
    persona = get_random_persona()
    prompt = f"create the most elaborate streamlined, hyper-casual web game inspired by the theme of a {persona}. The theme should only provide you with inspiration, they are not the end user. Your game should be fun and visually engaging above all. Once you have an initial idea or two, go back and make them more fun and engaging before executing on the best. Always include instuctions on how to play the game. The game will be played on PC never on mobile devices."

    # concurrently call all models
    tasks = []
    for model in reasoning_models:
        tasks.append(gen_reasoning(model, prompt, persona, client))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(run_reasoning_generation())
