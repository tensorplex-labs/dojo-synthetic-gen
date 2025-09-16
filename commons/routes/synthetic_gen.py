import asyncio
import functools
import json

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from commons.cache import RedisCache
from commons.synthetic import (
    run_dojo_v2_process,
    v2_get_augment_questions,
    v2_run_order_answer,
)
from commons.worker import WorkerManager

synthetic_gen_router = APIRouter(prefix="/api", tags=["synthetic-gen"])
cache = RedisCache()
worker = WorkerManager(
    do_work=functools.partial(
        run_dojo_v2_process,
    )
)


class SyntheticGenResponse(BaseModel):
    success: bool
    body: dict | None = None
    error: str | None = None


class QuestionResponse(BaseModel):
    success: bool
    prompt: str
    qa_id: str
    ans_aug_id: str


class AnswerResponse(BaseModel):
    success: bool
    ans_id: str


class AugmentQuestionResponse(BaseModel):
    success: bool
    augments: list


class OrderAnswerRequest(BaseModel):
    question: str


class AugmentQuestionRequest(BaseModel):
    question: str
    num_augments: int


class AnswerRequest(BaseModel):
    qa_id: str


@synthetic_gen_router.get(
    "/synthetic-gen",
    response_model=SyntheticGenResponse,
    summary="legacy v1 endpoint - returns random 1 qa pair from redis",
)
async def generate_synthetic_data():
    """
    v1 endpoint.
    """
    try:
        num_elems = await cache.get_queue_length()
        if num_elems == 0:
            for _ in range(100):
                await asyncio.sleep(3)
                num_elems = await cache.get_queue_length()
                if num_elems > 0:
                    break
            else:
                raise Exception("Cache population timeout after 300 seconds")

        qa_pair = await cache.dequeue()
        if qa_pair is None:
            raise Exception("Failed to get QA pair from cache")
        try:
            result = json.loads(qa_pair)
        except json.JSONDecodeError:
            result = {}

        return SyntheticGenResponse(success=True, body=result, error=None)
    except Exception as e:
        return SyntheticGenResponse(success=False, body={}, error=str(e))


@synthetic_gen_router.get(
    "/generate-question",
    response_model=QuestionResponse,
    summary="v2 - returns 1 x question from redis",
)
async def get_question():
    """
    v2 endpoint
    - returns questions that exist in questions redis queue.
    @returns {1 x question, 1 x uid}
    """
    try:
        question_payload = await cache.get_question()
        return QuestionResponse(
            success=True,
            prompt=question_payload["question"],
            qa_id=question_payload["qa_id"],
            ans_aug_id=question_payload["ans_aug_id"],
        )
    except Exception as e:
        logger.error(f"Error get_question: {e}")
        return QuestionResponse(success=False, prompt="", qa_id="", ans_aug_id="")


@synthetic_gen_router.post(
    "/generate-answer",
    response_model=AnswerResponse,
    summary="v2 - returns 1 x answer from redis",
)
async def get_answer(request: AnswerRequest):
    """
    v2 endpoint.
    -Returns a code answer stored with the input qa_id.
    @param qa_id: the redis key used to fetch the answer
    @return 1 answer stored retrieved from redis answers queue.
    @ to-do: add error handling when answer doesnt exist because it failed to generate.
    """
    try:
        answer_payload = await cache.get_answer(request.qa_id)
        if answer_payload is None:
            raise Exception("Failed to get answer from cache")
        return AnswerResponse(success=True, ans_id=request.qa_id)
    except Exception:
        return AnswerResponse(success=False, ans_id=request.qa_id)


@synthetic_gen_router.post("/get-question-augment")
async def get_question_augment(request: AugmentQuestionRequest):
    """
    v2 endpoint.
    - Requests for num_augments number of augmented questions.
    @param num_augments: the number of augments to generate must be between 1 and 3.
    @param base_question: the base question to augment
    @return augment_ids: the redis keys used to retrieve the augmented questions.
    """
    assert 1 <= request.num_augments <= 3, "num_augments must be between 1 and 3"
    try:
        augment_ids = await v2_get_augment_questions(
            request.question, request.num_augments
        )
        return AugmentQuestionResponse(success=True, augments=augment_ids)
    except Exception:
        return AugmentQuestionResponse(success=False, augments=[])


@synthetic_gen_router.post("/order-answer")
async def order_answer(request: OrderAnswerRequest):
    """
    v2 endpoint
    - bespoke function to generate code answers with any prompt.
    - intended to be used to generate with augmented prompts.
    @param question: the question to generate an answer for.
    @return answer_id: the redis key used to retrieve the answer.
    """
    try:
        answer_id = await v2_run_order_answer(request.question)
        return AnswerResponse(success=True, ans_id=answer_id)
    except Exception as e:
        logger.error(f"Error order_answer: {e}")
        return AnswerResponse(success=False, ans_id="")
