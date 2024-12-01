from pydantic import BaseModel, Field
from enum import Enum


class IELTSScoreEnum(Enum):
    BAND_1 = 1
    BAND_2 = 2
    BAND_3 = 3
    BAND_4 = 4
    BAND_5 = 5
    BAND_6 = 6
    BAND_7 = 7
    BAND_8 = 8
    BAND_9 = 9


class StructuredOutputSchema(BaseModel):
    ielts_band_score: IELTSScoreEnum = Field(
        default=IELTSScoreEnum.BAND_5, description="An IELTS band score ranging from 1 (non-user) to 9 (expert user)."
    )
