# from langfuse import langfuse_context
from typing import Dict

from langfuse import get_client


def _get_llm_usage(completion):
    usage: Dict[str, int] = {
        # "unit": "TOKENS",
        "input": int(completion.usage.prompt_tokens),
        "output": int(completion.usage.completion_tokens),
        "total_cost": int(completion.usage.total_tokens),
        "total": int(completion.usage.total_tokens),
        "input_cost": int(completion.usage.prompt_tokens),
        "output_cost": int(completion.usage.completion_tokens),
    }
    return usage


def log_to_langfuse(kwargs: dict, response_model, raw_response):
    """
    updates langfuse observation for the given kwargs
    @dev kwargs.pop() will modify the original kwargs dict. If we need the original kwargs, make a copy.
    """
    langfuse = get_client()
    kwargs["response_model"] = kwargs["response_model"].model_json_schema()
    langfuse.update_current_generation(
        input=kwargs.pop("messages"),
        model=kwargs.pop("model"),
        output=response_model.model_dump(),
        usage_details=_get_llm_usage(raw_response),
        metadata={
            **kwargs,
        },
    )
