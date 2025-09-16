import random

from commons.augmenter.types import (
    AnswerAugmentation,
    PAugmentation,
    QuestionAugmentation,
)
from commons.types import CodeAnswer, Topics


def _build_question_augment_prompt(
    base_question: str,
    augmentation: QuestionAugmentation,
    topic: Topics | None = None,
) -> str:
    # dirty hack for v2 augment question.
    if topic is None:
        topic = Topics.ANIMATION

    if augmentation == QuestionAugmentation.ADD_ONE:
        augment = f"Add one requirement to the question. Ensure your new requirement is distinct from the existing. Ensure that your new requirement does not break the functionality of the remaining requirements. Here is the generated coding question: {base_question}"
    elif augmentation == QuestionAugmentation.ADD_TWO:
        augment = f"Add two requirements to the question. Ensure your new requirements are distinct from the existing. Ensure that your new requirements do not break the functionality of the remaining requirements. Here is the generated coding question: {base_question}"
    elif augmentation == QuestionAugmentation.CHANGE_ANIMATION_OBJECT:
        if topic == Topics.SCIENCE:
            augment = f"Generate a new coding question similar to the original, but with a similar science experiment that is different from the original. Here is the original coding question: {base_question}"
        else:
            augment = f"Change the subject of the question to a different related subject such that rest of the question does not need to be modified. The new subject should be distinct from the original one, yet share enough characteristics such that the requirements still make sense. ie. If the original subject is a house with a requirements of windows, the new subject should be something that could feasibly also have windows. The new subject should be as similar to the original as possible, whilst still being distinguishable. As much as possible, please retain the requirements of the question. Here is the original coding question: {base_question}."
    elif augmentation == QuestionAugmentation.ORIGINAL:
        return base_question

    prompt = f"""
        <system>
            You are an LLM specializing in modifying existing coding questions to create similar yet distinct versions. Ultimately the questions that you generate will be implemented by a programming agent. As such, use your vast knowledge of UX and software engineering principles to make intelligent yet distinguishable modifications. Your response must only contain the modified question. Do not greet or converse with the user.
        </system>

        <user>
            {augment}
        </user>
    """
    return prompt


def _build_answer_augment_prompt(
    base_answer: CodeAnswer,
    base_question: str,
    augmentation: AnswerAugmentation,
) -> str:
    """
    creates the prompt to augment a base CodeAnswer
    introduces random chance for multiple augments to be applied at once.
    """

    answer_format = CodeAnswer.model_json_schema()
    augment = ""

    style_augment = "modify the colour scheme and font used in <base_answer>. The colour and font changes must be unrelated from the context identified in <question>."
    if augmentation == AnswerAugmentation.STYLE:
        augment = style_augment
    if augmentation == AnswerAugmentation.UX:
        ux_augment = "modify the user interactions in <base_answer>. The new interactions must be distinct from any user actions in <question>. Keep the modifications simple yet distinct."
        if random.randint(0, 1) == 0:
            augment = ux_augment
        else:
            augment = f"""{ux_augment} You must also {style_augment}"""
    if augmentation == AnswerAugmentation.ERROR:
        error_augment = "modify <base_answer> so a user action specified in <question> throws an exception instead of their current action."
        if random.randint(0, 2) != 0:
            augment = error_augment
        else:
            augment = f"""{error_augment} You must also {style_augment}"""

    prompt = f"""
    <system>
        Here is the base HTML file with in-line Javascript code you must improve:
        <base_answer>
            {base_answer}
        </base_answer>
        Here are the specifications that were used to create the <base_answer>:
        <question>
            {base_question}
        </question>

        <response_format>
        your response must always be valid json based on this schema:
        {answer_format}
        </response_format>

        <role>
             You are an expert natural language coding agent, demonstrating how bad UX design choices can worsen the user experience.
             Your objective is to {augment}
        </role>
        <instructions>
            Always follow these instructions:
            - Your code must not contain any comments
            - You do not have access to the file system. Do not store any data in storage or as a file.
            - Ensure that your code does not use any external files such as images, videos or audio files.
            - Your code must not require the use of the user's microphone or camera.
            - Your code must not use any external libraries, data or APIs.
            - Your code must not modify the name of any existing headers or titles.
            - Ensure your output is complete and executable. Do not hide existing code for the users convenience.
        </instructions>
    </system>
    <user>
        {augment}
    </user>
    """
    return prompt


def _build_performance_augment_prompt(
    base_answer: CodeAnswer,
    base_question: str,
    augmentation: PAugmentation,
) -> str:
    """
    creates the prompt to augment a base CodeAnswer with performance issues.
    """

    answer_format = CodeAnswer.model_json_schema()
    augment = ""

    if augmentation == PAugmentation.AUGMENT_1:
        augment = "Modify the solution so that user interactions have a noticeable delay, making the UI feel sluggish. For example, add a 1-second delay to button clicks or hover effects."
    elif augmentation == PAugmentation.AUGMENT_2:
        augment = "Modify the solution to make animations appear 'janky' or not smooth. You could achieve this by using inefficient rendering techniques or high-frequency DOM manipulation inside a loop."
    elif augmentation == PAugmentation.AUGMENT_3:
        augment = "Modify the solution to introduce a visual glitch that occurs during user interaction. For example, an element might flicker, leave a trail when dragged, or temporarily disappear."

    prompt = f"""
    <system>
        Here is the base HTML file with in-line Javascript code you must modify:
        <base_answer>
            {base_answer}
        </base_answer>
        Here are the specifications that were used to create the <base_answer>:
        <question>
            {base_question}
        </question>

        <response_format>
        your response must always be valid json based on this schema:
        {answer_format}
        </response_format>

        <role>
             You are an expert natural language coding agent, demonstrating how bad performance choices can worsen the user experience.
             Your objective is to {augment}
        </role>
        <instructions>
            Always follow these instructions:
            - Your code must not contain html or javascript comments
            - Do not use words such as janky, glitch or something similar in the code
            - You do not have access to the file system. Do not store any data in storage or as a file.
            - Ensure that your code does not use any external files such as images, videos or audio files.
            - Your code must not require the use of the user's microphone or camera.
            - Your code must not use any external libraries, data or APIs.
            - Your code must not modify the name of any existing headers or titles.
            - Ensure your output is complete and executable. Do not hide existing code for the users convenience.
            - The core requirements of the original question should be preserved in your modified answer.
        </instructions>
    </system>
    <user>
        {augment}
    </user>
    """
    return prompt
