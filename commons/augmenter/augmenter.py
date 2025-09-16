"""
Augmenter generates necessary augments concurrently. Used by synthetic.py
"""

import asyncio
import random
import uuid

from instructor import AsyncInstructor

# from langfuse.decorators import observe
from langfuse import observe
from loguru import logger

from commons.augmenter.prompts import (
    _build_answer_augment_prompt,
    _build_performance_augment_prompt,
    _build_question_augment_prompt,
)
from commons.augmenter.types import (
    AnswerAugmentation,
    AugmentQuestionResponse,
    PAugmentation,
    QuestionAugmentation,
)
from commons.linter.linter import lint_and_fix_code
from commons.llm import call_llm
from commons.types import (
    CodeAnswer,
    CodeQuestion,
    GeneratedAnswer,
    Topics,
)
from commons.utils.logging import log_to_langfuse
from commons.utils.utils import reject_duplicate_ans_augment


class Augmenter:
    def __init__(self, client: AsyncInstructor, model: str):
        self.client: AsyncInstructor = client
        self.model = model

    async def _gen_augment(
        self,
        base_question: str,
        base_answer: CodeAnswer,
        rank: int,
        topic: Topics,
    ):
        """
        generate 1 augmented answer for a given ground truth rank.
        """
        from commons.synthetic import (
            generate_answer,  # lazy import to prevent circular import error
        )

        # randomly select an augment type based on the input rank
        selected_augment = random.choice(
            [QuestionAugmentation, AnswerAugmentation, PAugmentation]
        )(rank)
        try:
            if isinstance(selected_augment, AnswerAugmentation):
                augmented_answer = await self._augment_answer(
                    base_question, base_answer, selected_augment
                )
            elif isinstance(selected_augment, QuestionAugmentation):
                q = await self._augment_question(base_question, selected_augment, topic)
                augmented_answer = await generate_answer(
                    self.client, self.model, q.question, topic, rank
                )
            elif isinstance(selected_augment, PAugmentation):
                augmented_answer = await self._augment_performance(
                    base_question, base_answer, selected_augment
                )
        except Exception as e:
            raise e
        return augmented_answer

    async def run_unified_augment(
        self,
        base_question: str,
        base_answer: CodeAnswer,
        topic: Topics,
    ):
        """
        concurrently generate 3 augmentations, 1 for each ground truth rank
        """

        try:
            tasks = [
                self._gen_augment(
                    base_question,
                    base_answer,
                    rank,
                    topic,
                )
                for rank in range(1, 4)
            ]
            return await asyncio.gather(*tasks)
        except Exception as e:
            raise e

    @observe(as_type="generation", capture_input=True, capture_output=True)
    async def _augment_answer(
        self,
        base_question: str,
        base_answer: CodeAnswer,
        augmentation: AnswerAugmentation,
        retry: bool = False,
    ) -> GeneratedAnswer:
        """
        returns a single answer augmentation
        """

        # early return the original answer
        if augmentation == AnswerAugmentation.ORIGINAL:
            return GeneratedAnswer(
                model=self.model,
                augment=augmentation.value,
                answer=base_answer,
                id=str(uuid.uuid4()),
            )

        # @dev: id should identify a given output.
        id = str(uuid.uuid4())

        # build prompt
        messages = [
            {
                "role": "system",
                "content": _build_answer_augment_prompt(
                    base_answer, base_question, augmentation
                ),
            },
        ]

        # query LLM
        kwargs = {
            "response_model": CodeAnswer,
            "model": self.model,
            "messages": messages,
            "temperature": random.uniform(0, 0.1),
            "max_tokens": 64000,
            "top_p": random.uniform(0, 0.6),
            "response_format": {
                "type": "json_object",
                "response_schema": CodeAnswer.model_json_schema(),
                "enforce_validation": True,
            },
        }

        try:
            result, completion = await call_llm(self.client, kwargs)

            # log to langfuse
            kwargs["question"] = base_question
            kwargs["augmentation_level"] = augmentation
            log_to_langfuse(kwargs, result, completion)

            # apply linting and fix syntax errors
            result = await lint_and_fix_code(self.client, self.model, result, id)

            # check if generated code is same as base answer. If true then retry generation.
            if reject_duplicate_ans_augment(base_answer, result):
                if not retry:
                    logger.error(
                        f" {id} {augmentation} answer generated is same as base_answer. Retrying ..."
                    )
                    return await self._augment_answer(
                        base_question, base_answer, augmentation, retry=True
                    )
                # if unique augment is not generated then throw exception.
                elif retry:
                    logger.error(
                        f"{id} {augmentation} failed to generate unique augment"
                    )
                    raise Exception(
                        f"{id} {augmentation} failed to generate unique augment"
                    )

            logger.info(f" {id} {augmentation} answer generated")
            return GeneratedAnswer(
                model=self.model,
                augment=augmentation.value,
                answer=result,
                id=id,
            )
        except Exception as e:
            logger.error(f"{id} failed to generate augmented answer: {e}")
            raise e

    @observe(as_type="generation", capture_input=True, capture_output=True)
    async def _augment_performance(
        self,
        base_question: str,
        base_answer: CodeAnswer,
        augmentation: PAugmentation,
        retry: bool = False,
    ) -> GeneratedAnswer:
        """
        returns a single performance augmentation
        """
        # early return the original answer
        if augmentation == PAugmentation.ORIGINAL:
            return GeneratedAnswer(
                model=self.model,
                augment=augmentation.value,
                answer=base_answer,
                id=str(uuid.uuid4()),
            )

        id = str(uuid.uuid4())

        # build prompt
        messages = [
            {
                "role": "system",
                "content": _build_performance_augment_prompt(
                    base_answer, base_question, augmentation
                ),
            },
        ]

        # query LLM
        kwargs = {
            "response_model": CodeAnswer,
            "model": self.model,
            "messages": messages,
            "temperature": random.uniform(0, 0.1),
            "max_tokens": 64000,
            "top_p": random.uniform(0, 0.6),
            "response_format": {
                "type": "json_object",
                "response_schema": CodeAnswer.model_json_schema(),
                "enforce_validation": True,
            },
        }

        try:
            result, completion = await call_llm(self.client, kwargs)

            # log to langfuse
            kwargs["question"] = base_question
            kwargs["augmentation_level"] = augmentation
            log_to_langfuse(kwargs, result, completion)

            # apply linting and fix syntax errors
            result = await lint_and_fix_code(self.client, self.model, result, id)

            # check if generated code is same as base answer. If true then retry generation.
            if reject_duplicate_ans_augment(base_answer, result):
                if not retry:
                    logger.error(
                        f" {id} {augmentation} answer generated is same as base_answer. Retrying ..."
                    )
                    return await self._augment_performance(
                        base_question, base_answer, augmentation, retry=True
                    )
                # if unique augment is not generated then throw exception.
                elif retry:
                    raise Exception(
                        f"{id} {augmentation} failed to generate unique augment"
                    )

            logger.info(f" {id} {augmentation} answer generated")
            return GeneratedAnswer(
                model=self.model,
                augment=augmentation.value,
                answer=result,
                id=id,
            )
        except Exception as e:
            logger.error(f"{id} failed to generate augmented answer: {e}")
            raise e

    @observe(as_type="generation", capture_input=True, capture_output=True)
    async def _augment_question(
        self,
        question: str,
        augmentation_level: QuestionAugmentation,
        topic: Topics | None = None,
        retry: bool = False,
    ) -> AugmentQuestionResponse:
        """Augment the question with the given model and augmentation level."""

        # early return the original question.
        if augmentation_level == QuestionAugmentation.ORIGINAL:
            return AugmentQuestionResponse(
                question=question,
                augmentation_level=augmentation_level,
            )

        kwargs = {
            "response_model": CodeQuestion,
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": _build_question_augment_prompt(
                        question, augmentation_level, topic
                    ),
                }
            ],
            "temperature": random.uniform(0, 1),
            "max_tokens": 8192,
            "top_p": random.uniform(0, 0.6),
        }

        if self.model.startswith("openai"):
            kwargs["seed"] = random.randint(0, int(1e9))  # needed for OpenAI
        try:
            response_model, completion = await call_llm(self.client, kwargs)

            # retry if augmented question is under 100 tokens
            if completion.usage.completion_tokens < 100:
                if not retry:
                    return await self._augment_question(
                        question, augmentation_level, topic, retry=True
                    )
                else:
                    raise Exception(
                        "failed to augment question over 100 tokens after max retries"
                    )

            # logging
            kwargs["question"] = question
            kwargs["augmentation_level"] = augmentation_level
            log_to_langfuse(kwargs, response_model, completion)
            logger.info(f"{augmentation_level} Completed")
            return AugmentQuestionResponse(
                question=response_model.question,
                augmentation_level=augmentation_level,
            )
        except Exception as e:
            logger.error(f"{augmentation_level}: failed to augment question: {e}")
            raise

    async def v2_run_augment_question(self, base_question: str, num_augments: int):
        """
        - concurrently generate num_augments augmented questions.
        - num_augments must be between 1 and 3.
        """
        assert 1 <= num_augments <= 3, "num_augments must be between 1 and 3"
        # select random augment types
        available_types = [QuestionAugmentation(i) for i in range(3)]
        selected_types = random.sample(available_types, num_augments)

        tasks = []
        try:
            for augment_type in selected_types:
                tasks.append(self._augment_question(base_question, augment_type))
            return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error v2_run_augment_question: {e}")
            raise e
