import asyncio
import json
from collections.abc import Awaitable
from datetime import datetime
from typing import Any, cast

import uuid_utils
from fastapi.encoders import jsonable_encoder
from loguru import logger
from redis import asyncio as aioredis
from redis.asyncio.client import Redis

from commons.config import RedisSettings, get_settings, parse_cli_args


def build_redis_url() -> str:
    redis: RedisSettings = get_settings().redis
    if redis.username and redis.password:
        return f"redis://{redis.username}:{redis.password.get_secret_value()}@{redis.host}:{redis.port}"
    elif redis.password:
        return f"redis://:{redis.password.get_secret_value()}@{redis.host}:{redis.port}"
    else:
        return f"redis://{redis.host}:{redis.port}"


class RedisCache:
    _instance: "RedisCache | None" = None
    _key_prefix: str = (
        "synthetic"  # all entries from synthetic-gen should be prefixed with this
    )
    _queue_key: str = "queue"
    _hist_key_prefix: str = "history"  # key prefix to historical data
    _num_workers_active_key: str = (
        "num_workers_active"  # key to figure out how many workers are working
    )
    _question_key = "questions"
    _answer_key = "answers"
    _qn_augment_key = "qn_augments"
    _human_feedback_key_prefix: str = "hf"
    _encoding: str = "utf-8"
    redis: Redis  # pyright: ignore[reportMissingTypeArgument]

    def __new__(cls) -> "RedisCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            redis_url = build_redis_url()
            cls._instance.redis = aioredis.from_url(url=redis_url)
        return cls._instance

    def _build_key(self, *parts: str) -> str:
        if len(parts) == 0:
            raise ValueError("Must specify at least one redis key")
        return f"{self._key_prefix}:{':'.join(parts)}"

    async def close(self) -> None:
        try:
            # clear all active workers
            delta = -1 * await self.get_num_workers_active()
            await self.update_num_workers_active(delta)

            if self.redis:
                await self.redis.close()
        except Exception as exc:
            logger.opt(exception=True).error(f"Error closing Redis connection: {exc}")

    async def get_queue_length(self) -> int:
        key = self._build_key(self._question_key)
        num_items: int = await cast(Awaitable[int], self.redis.llen(key))
        logger.trace(f"Queue length: {num_items}, time: {(datetime.now().timestamp())}")
        return num_items

    async def get_num_workers_active(self) -> int:
        key = self._build_key(self._num_workers_active_key)
        value = await self.redis.get(key)
        num_active = 0 if value is None else int(value)
        logger.trace(
            f"Number of active workers: {num_active}, time: {(datetime.now().timestamp())}"
        )
        return num_active

    async def update_num_workers_active(self, delta: int) -> int:
        """Update the number of workers active by delta.

        Args:
            delta (int): Amount to increment/decrement the count by. To decrement use a negative number.

        Returns:
            int: The new number of workers active.
        """
        key = self._build_key(self._num_workers_active_key)

        lock_key = self._build_key(key, "lock")
        num_workers_active_lock = self.redis.lock(
            name=lock_key,
            timeout=60,
            blocking=True,
        )
        num_active: int = 0
        try:
            if await num_workers_active_lock.acquire():
                num_active = await self.get_num_workers_active() + delta
                # ensure it doesn't go below 0
                num_active = max(num_active, 0)
                await self.redis.set(key, num_active)
        finally:
            # ensure we always release the lock
            try:
                await num_workers_active_lock.release()
            except Exception:
                pass

        return num_active

    async def enqueue(self, data: Any) -> int:
        """Uses Redis list to enqueue data, in order to maintain a buffer of QA
        pairs. This is because each QA pair may take long to generate and we
        want to maintain responsiveness of the FastAPI app.

        Args:
            key (str): Redis key that will be prefixed with `_key_prefix`.
            data (Any): Data to be enqueued.

        Raises:
            ValueError: If data is None.

        Returns:
            int: Number of elements in the queue.
        """

        if data is None:
            raise ValueError("Data is required")

        # keep the historical data as is
        # use uuid7 so keys in redis are sorted by time
        redis_task_id = uuid_utils.uuid7().__str__()
        hist_key = self._build_key(self._hist_key_prefix, redis_task_id)
        try:
            logger.debug(f"Writing persistent data into {hist_key}")
            # place into persistent key
            str_data = json.dumps(jsonable_encoder(data)).encode(self._encoding)
            args = parse_cli_args()
            if args.env_name and args.env_name == "prod":
                # expire in 4 hours time
                await self.redis.set(hist_key, str_data, ex=3600 * 4)  # pyright: ignore[reportUnknownMemberType]
            else:
                await self.redis.set(hist_key, str_data)  # pyright: ignore[reportUnknownMemberType]

            queue_key = self._build_key(self._queue_key)
            logger.debug(f"Writing queue data into {queue_key}")
            # fuck it and push the data into the queue as well, instead of it being a reference to the persistent data
            # this also simplifies dequeuing logic
            num_items: int = await self.redis.rpush(queue_key, str_data)  # type: ignore

            # collect cids for each answer and log successful upload to DB
            ids: list[str] = [response["cid"] for response in data["responses"]]
            logger.success(f"Pushed Task {ids} to DB")
            return num_items
        except Exception as exc:
            logger.opt(exception=True).error(
                f"Error enqueuing data into key: {hist_key}, error: {exc}"
            )
            raise

    async def dequeue(self) -> str | None:
        """Dequeue an item from the specified queue key, assumes it is a queue and
        returns the value as as a string
        """
        current_key: str = self._build_key(self._queue_key)
        try:
            value_raw = await cast(
                Awaitable[str | bytes | list[Any] | None],
                self.redis.lpop(current_key),  # pyright: ignore[reportUnknownMemberType]
            )

            value: str | None = None
            if isinstance(value_raw, list):
                raise NotImplementedError("not implemented")
            elif isinstance(value_raw, bytes):
                value = value_raw.decode(self._encoding)
            return value
        except Exception as exc:
            logger.opt(exception=True).error(
                f"Error dequeuing data from key: {current_key}, error: {exc}"
            )
            raise

    async def store_question(self, qa_id: str, question: str, ans_aug_id: str):
        """
        stores generated questions in redis question queue.
        """
        key = self._build_key(self._question_key)
        q_payload = {
            "question": question,
            "qa_id": qa_id,
            "ans_aug_id": ans_aug_id,
        }
        str_data = json.dumps(jsonable_encoder(q_payload)).encode(self._encoding)
        try:
            await self.redis.rpush(key, str_data)
            logger.trace(f"Stored question {qa_id} in redis")
        except Exception as e:
            logger.error(f"Error adding prompt {qa_id} to redis, error: {e}")
            raise

    async def get_question(self):
        try:
            """
            pops element from question queue.
            """
            key = self._build_key(self._question_key)
            # @DEV: dont pop from redis when dev flag is used - remove this in prod pls pls
            # args = parse_cli_args()
            # question_payload = {}
            # if args.env_name and args.env_name == "dev":
            question = await self.redis.lindex(key, 0)
            # else:
            #     question = await self.redis.lpop(key)
            #     if question:
            #         question_payload = json.loads(question)  # type: ignore
            #         qa_id = question_payload.get("qa_id")
            #         ans_aug_id = question_payload.get("ans_aug_id")
            #         await self.redis.delete(self._build_key(self._answer_key, qa_id))
            #         await self.redis.delete(
            #             self._build_key(self._answer_key, ans_aug_id)
            #         )
            return json.loads(question)  # type: ignore
        except Exception as e:
            logger.error(f"Error get_question from redis, error: {e}")
            raise

    async def store_answer(self, answer_payload: dict):
        try:
            """
            - stores generated answers in redis answers queue.
            """
            qa_id = answer_payload["qa_id"]
            key = self._build_key(self._answer_key, qa_id)
            answer_data = json.dumps(jsonable_encoder(answer_payload)).encode(
                self._encoding
            )
            await self.redis.set(key, answer_data)
            logger.trace(f"Stored answer {qa_id} in redis")
        except Exception as e:
            logger.error(f"Error storing answer in redis, error: {e}")
            raise

    async def get_answer(self, qa_id: str):
        """
        gets answer from redis answers queue.
        """
        try:
            key = self._build_key(self._answer_key, qa_id)
            answer = await self.redis.get(key)
            return answer
        except Exception as e:
            logger.error(f"Error getting answer from redis {e}")
            raise

    async def store_augmented_question(self, augmented_question: str, augment_id: str):
        """
        stores augmented questions in qn_augments queue.
        """
        key = self._build_key(self._qn_augment_key, augment_id)
        augmented_question_data = json.dumps(
            jsonable_encoder(augmented_question)
        ).encode(self._encoding)
        try:
            await self.redis.set(key, augmented_question_data)
            logger.trace(f"Stored augmented question {augment_id} in redis")
        except Exception as e:
            logger.error(f"Error storing augmented question in redis, error: {e}")
            raise

    async def remove_qa_by_id(self, qa_id: str) -> int:
        """
        removes the given question, its answer, and its augmented answer from redis.
        """
        key = self._build_key(self._question_key)
        try:
            elements: list[bytes] = await self.redis.lrange(key, 0, -1)

            element_to_delete = None
            aug_id = None
            for element_bytes in elements:
                element_str = element_bytes.decode(self._encoding)
                try:
                    payload = json.loads(element_str)
                    if payload.get("qa_id") == qa_id:
                        element_to_delete = element_bytes
                        aug_id = payload.get("ans_aug_id")
                        break
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode element in list {key}: {element_str}"
                    )
                    continue

            if element_to_delete:
                num_removed: int = await self.redis.lrem(key, 1, element_to_delete)
                if num_removed > 0:
                    logger.info(f"Deleted question with qa_id {qa_id} from the queue.")
                    await self.redis.delete(self._build_key(self._answer_key, qa_id))
                    if aug_id:
                        await self.redis.delete(
                            self._build_key(self._answer_key, aug_id)
                        )
                return num_removed
            else:
                logger.info(f"Question with qa_id {qa_id} not found in the queue.")
                return 0
        except Exception as e:
            logger.error(
                f"Error deleting question with qa_id {qa_id} from redis, error: {e}"
            )
            raise


# for testing.
# to run:
#   python -m commons.cache.redis --env_name dev
async def main():
    r = RedisCache()
    # await r.store_question("1", "prompt3")
    # await r.store_question("2", "prompt4")
    question = await r.get_question()
    print(question)


if __name__ == "__main__":
    asyncio.run(main())
