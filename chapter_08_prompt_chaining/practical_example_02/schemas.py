from pydantic import BaseModel, Field
from enum import Enum


class PerformanceCategory(str, Enum):
    excellent = "Excellent"
    good = "Good"
    fair = "Fair"
    poor = "Poor"


class Criterion(BaseModel):
    feedback: str = Field(description="Descriptive feedback or reasoning for the grade.")
    category: PerformanceCategory = Field(description="Performance classification: Excellent, Good, Fair, or Poor.")
    score: float = Field(description="Numerical score for the criterion, ranging from 0.0 to 100.0.")
    meets_standard: bool = Field(description="Boolean indicating if the criterion meets the expected standard.")


class StructuredOutputSchema(BaseModel):
    content: Criterion = Field(description="Evaluation details for Content.")
    clarity: Criterion = Field(description="Evaluation details for Clarity.")
    research: Criterion = Field(description="Evaluation details for Research.")
    presentation: Criterion = Field(description="Evaluation details for Presentation.")