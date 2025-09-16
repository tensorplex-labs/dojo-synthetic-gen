import Levenshtein
from loguru import logger

from commons.types import CodeAnswer


def get_js_from_code_answer(answer: CodeAnswer) -> str:
    _, js_file = next(
        (i, file) for i, file in enumerate(answer.files) if file.filename == "index.js"
    )
    return js_file.content


def get_html_from_code_answer(answer: CodeAnswer) -> str:
    _, html_file = next(
        (i, file)
        for i, file in enumerate(answer.files)
        if file.filename == "index.html"
    )
    return html_file.content


def reject_duplicate_ans_augment(
    base_answer: CodeAnswer, new_answer: CodeAnswer
) -> bool:
    """
    given two CodeAnswers, rejects if their composite HTML and JS are more than 98% similar.
    """
    base_js = get_js_from_code_answer(base_answer)
    new_js = get_js_from_code_answer(new_answer)

    base_html = get_html_from_code_answer(base_answer)
    new_html = get_html_from_code_answer(new_answer)

    base_content = base_html + base_js
    new_content = new_html + new_js

    leven_similarity = Levenshtein.ratio(base_content, new_content)
    logger.info(f"Levenshtein similarity: {leven_similarity}")
    return leven_similarity > 0.99
