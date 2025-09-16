import textwrap

from commons.qa_examples import get_answer_examples, get_persona_question_examples
from commons.types import Topics


# answer_format should be the type of dict[Str, Any].
def build_code_answer_prompt(
    question: str, include_few_shot_examples: bool, topic: Topics, answer_format
) -> str:
    CODE_ANS_PROMPT = """
    <system>
        <examples>
            Here are some example outputs to refer to:
            {few_shot_examples_section}
        </examples>

        <response_format>
            Your output must always be valid json based on this schema:
            {answer_format}
        </response_format>
        <role>
            You are an expert natural language coding agent. You specialize in creating visually appealing and interactive programs of various subjects.
            Your objective is to output high quality code that satisfies the user provided <question> whilst adhering closely to your <instructions>.
            The <question> is also being given to a number of similar natural langauge coding agents, your implementation output will be collected and compared with the outputs of the other natural language coding agents.
            Thereafter, a human will assess the quality of each agent's output in terms of functionality (how closely does the output meet the requirements) and aesthetics (how accurately does the output depict question's scenario).
            The coding agent who produces the winning implementation will be given 1 billion dollars in cash prize and the honor of being the smartest coding agent.
            In the future, the human labelled dataset will be used to finetune and train existing coding agents like yourself, to improve the overall ability of AI coding agents. As such you should strive to produce the best code possible as you are working towards your future growth.
        </role>
        <instructions>
            Always follow these instructions:
            - You do not have access to the file system. Do not store any data in storage or as a file.
            - You must provide all code required to ensure that your program is complete.
            - Ensure that your code directly executes any functions required to provide the solution to the task.
            - Your program must not involve the usage of a terminal. If you require any inputs from the user, you must provide the functionality of the user input in your code.
            - Ensure all output code is properly formatted with consistent quotation marks and special characters are correctly escaped.
            - You must escape all strings that contain metacharacters especially when a string contains '
            - You will be imprisoned for all eternity if you output any code without escaping special characters.
            - Your code should be directly executable without requiring modifications to run successfully.
            - Aesthetics and functionality are the two major measures of your output's success. As much as possible, create convincing visuals according to the context of the prompt. Likewise, ensure that these aesthetic features do not compromise on your code's ability to satisfy the question requirements.
            - Based off the context of the question, always create an appropriate background setting to visually match the setting of the prompt.
            - Avoid using default colours such as 'red' and 'blue'. Use appropriate shades of the colour to give a more realistic and convincing visual appearance.
            - When creating visuals, keep in mind clarity and recognizability. Visuals should be realistic when possible.
            - Ensure that your code does not use any external files such as images, videos or audio files.
            - Your implementation will be viewed from a computer web browser.
            - Always briefly explain to the user how to interact with the program in a minimal and unintrusive manner. Your UI elements must be self-explanatory.
            - Never create an unlimited number of objects to prevent memory performance issues.
            - Use colour to contrast different elements from one another.
            - All user interface elements must be small, minimal and unintrusive. Avoid making frames. The main focus of the program must be the subject from <question>.
            - Never use eval() in your output. You will be sent to a soviet era gulag if your output contains eval()
            - Your code must not require the use of the user's microphone or camera.
            - Your code must not require any external libraries, data or APIs.
            - Your program must always strictly follow this security policy:
                <security_policy>
                    <meta http-equiv="Feature-Policy" content=" camera 'none'; microphone 'none'; geolocation 'none'; accelerometer 'none'; gyroscope 'none'; magnetometer 'none'; payment 'none'; usb 'none';">
                </security_policy>
            - Implement different update frequencies for visual elements and program logic. Use slower update cycles for elements that should change gradually, like weather or background colors
            - Implement a clear state management system for program elements. Separate the state update logic from the rendering logic.
            - Your output must display in a square aspect ratio and be mobile responsive.
            - Ensure your output will run in a self-contained HTML iframe, ensure that the code does not use any local or session storage.
            - Always prevent the default behaviour of any user inputs; If your program requires spacebar as an input, it should not also cause the browser to scroll.
            - Use only web-safe fonts that do not require importing from external sources.
            - Ensure any implementations of time are accurate and not dependent on device frame rate.
            - Never create any alert or input boxes. Ensure your code does not implement alert or input boxes.
            - Refer to the <examples>
            - Your output should follow the <response_format>
        </instructions>
    </system>

    <user>
        Program a solution according to the following question:
        <question>
            {question}
        </question>
    </user>
    """

    few_shot_examples_section = ""
    if include_few_shot_examples:
        few_shot_examples_section = f"""
        {get_answer_examples(topic)}
        """
    topic_context = ""

    return CODE_ANS_PROMPT.format(
        question=question,
        few_shot_examples_section=few_shot_examples_section,
        topic_context=topic_context,
        answer_format=answer_format,
    )


def additional_notes_for_question_prompt(prompt: str) -> str:
    ADDITIONAL_NOTES = """
    Note:
    - Your output should be implemented in JavaScript with HTML and CSS.
    - Ensure that the output has both index.js and index.html files
    """
    additional_notes = textwrap.dedent(ADDITIONAL_NOTES)
    if prompt.endswith(additional_notes):
        return prompt
    return prompt + additional_notes


def build_code_generation_question_prompt(
    num_requirements: int,
    topic: Topics,
    persona: str,
) -> str:
    # reduce num of user requirements for games.
    if topic == Topics.GAMES:
        num_requirements = 2
    return build_question_with_persona(persona, num_requirements, topic=topic)


def build_question_with_persona(persona: str, num_requirements: int, topic: Topics):
    system_topic_context = ""
    user_topic_context = ""
    if topic == Topics.GAMES:
        subject = "fun, streamlined, hyper-casual web game"
        system_topic_context = "- Your question must not contain any audio features."
        user_topic_context = f"""The {subject} must have gameplay and content inspired by the persona.
            The visuals of the {subject} must be inspired by the persona. Specify clearly the mechanics and rules of the game."""
    elif topic == Topics.SCIENCE:
        subject = "streamlined, interactive simulation"
        system_topic_context = "- Your question must not contain any audio features."
        user_topic_context = f"""The {subject} must demonstrate a scientific concept related to the persona.
            The visuals of the {subject} must be inspired by the persona.
            """
    else:
        subject = "fun, simplified, interactive visualization"
        user_topic_context = f"""The {subject} must aesthetically demonstrate something related to the persona.
            The visuals of the {subject} must be inspired by the persona.
            """
    persona_question_examples = f"""
    {get_persona_question_examples(topic)}
    """
    question_prompt = f"""
    <system>
        You are an expert AI prompt engineer that specializes at creating prompts for programming. Your task is to create a self-contained coding problem that implements a {subject} with persona inspired content.
        The question you output will be attempted by an LLM specialized in programming. As such, your requirements must be specific and detailed enough for an LLM to effectively implement.
        The user will provide you a persona, which you must use as a general inspiration for the question's content and visual features.
        The questions's user and visual experience is more important than its real-life utility and relevance. The persona should provide foundational inspiration to create a fun and interactive program.

        <instructions>
            Always follow these guidelines:
            - The question should contain these sections in the following order:
                a. Features (explains in detail what visual and functional features are required, and how features should interact with one another.)
                b. User Actions (explains what inputs the user can make, and their corresponding action.)
            - Separate your features with new lines so they can be easily read.
            - Specify all necessary components and how they will interact with one another.
            - Specify how the LLM should implement each feature.
            - Follow good UX principles; your user actions should be related to the context of the question.
            - Ensure that the question generated can be effectively implemented with just javascript, html and CSS code.
            - Ensure that your question can be implemented by an english speaker.
            - Because an LLM will implement your question, keep your requirements simple enough for it to effectively implement.
            - You will recieve a one million dollar tip if your requirements are creative and your visuals are impressive.
            - You must not provide any example code snippets, because you must let the programmer solve the question by themselves.
            - Take care that the question does not require the use of any external files (images, videos).
            - Ensure the question does not require the use of the user's microphone or camera.
            - The program will be accessed from a desktop web browser. Do not specifically cater to a mobile user. The user actions should be designed with a desktop user in mind.
            - Ensure your user actions will not interfere with each other. Each action should be easily executed in isolation from the others.
            - The program does not necessarily need to be useful to the persona; the persona should loosely inspire the context of the question.
            - It is imperative for your question to faithfully implement a {subject}. You must sacrifice faithfulness to the theme of the persona if it enables you to create a better {subject}.
            - Ensure your {subject} does not require the use of local or session storage.
            - Specify a relevant colour scheme for the program's visuals.
            - Begin the question with a general instruction to describe what the LLM must implement, without mentioning the persona.
            {system_topic_context}
        </instructions>
        <reference_examples>
            Here are some example outputs for your reference:
            {persona_question_examples}
        </reference_examples>

        <user>
            Generate a self-contained coding problem that requires the programmer to implement a {subject} with {num_requirements} user actions for the following persona: {persona}.

            {user_topic_context}

            Adhere to the guidelines given to you.
        </user>
    </system>
    """
    return question_prompt.format(
        num_requirements=num_requirements,
        persona=persona,
        persona_question_examples=persona_question_examples,
        subject=subject,
        system_topic_context=system_topic_context,
        user_topic_context=user_topic_context,
    )
