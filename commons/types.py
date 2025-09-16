from dataclasses import dataclass
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Topics(Enum):
    ANIMATION = 0
    # LANDSCAPES = 1
    SCIENCE = 1
    # THREE_D = 3
    GAMES = 2


class FileObject(BaseModel):
    filename: str = Field(description="Name of the file")
    content: str = Field(description="The code contents of the file.")


class CodeAnswer(BaseModel):
    files: List[FileObject] = Field(
        description="Array of FileObject, that are part of the code solution. Must include index.html, and index.js a Javascript solution"
    )


class CodeQuestion(BaseModel):
    question: str = Field(
        description="Coding question to be solved by a software engineer"
    )


@dataclass
class GeneratedAnswer:
    id: str
    model: str
    augment: int | None
    answer: CodeAnswer
