from dataclasses import dataclass
from enum import Enum


class QuestionAugmentation(Enum):
    ORIGINAL = 0
    ADD_TWO = 1
    ADD_ONE = 2
    CHANGE_ANIMATION_OBJECT = 3


class AnswerAugmentation(Enum):
    ORIGINAL = 0
    STYLE = 1
    UX = 2
    ERROR = 3


class PAugmentation(Enum):
    ORIGINAL = 0
    AUGMENT_1 = 1  # Unresponsive UI
    AUGMENT_2 = 2  # Inefficient Animation
    AUGMENT_3 = 3  # Visual Glitch


@dataclass
class AugmentQuestionResponse:
    question: str
    augmentation_level: QuestionAugmentation
