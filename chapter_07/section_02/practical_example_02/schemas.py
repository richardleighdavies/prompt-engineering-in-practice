""" """

# Standard Library
from enum import Enum

# Third Party
from pydantic import BaseModel, Field


class RatingLevel(Enum):
    """Rating levels"""
    POOR = "Poor"
    FAIR = "Fair"
    GOOD = "Good"
    VERY_GOOD = "Very Good"
    EXCELLENT = "Excellent"


class OutputSchema(BaseModel):
    """Synthesized product review analysis"""
    
    product_name: str = Field(description="The official name of the product as it appears in the marketplace.")
    summary: str = Field(description="A concise overview of the product's key features, performance, and overall assessment.")
    
    # Core analysis
    key_points: dict[str, list[str]] = Field(description="Structured collection of the product's strengths and weaknesses organized by category.")
    market_analysis: str = Field(description="Assessment of how the product compares to competitors and its position within its target market.")
    
    # Conclusions
    recommendation: str = Field(description="Definitive buying advice including ideal use cases and potential customer profiles.")
    overall_rating: int = Field(description="Numerical evaluation of the product's overall quality on a scale from 1 (worst) to 10 (best).", ge=1, le=10)
    rating_category: RatingLevel = Field(description="Qualitative assessment of the product from Poor to Excellent that corresponds with the numerical rating.")
