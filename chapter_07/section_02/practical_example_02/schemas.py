""" 
Schema definitions for the Product Review Synthesizer.
These schemas define the structure for the various analysis outputs.
"""

# Standard Library
from enum import Enum

# Third Party
from pydantic import BaseModel, Field


class RatingLevel(Enum):
    """Enum for standardized rating levels"""

    POOR = "Poor"
    FAIR = "Fair"
    GOOD = "Good"
    VERY_GOOD = "Very Good"
    EXCELLENT = "Excellent"


class FeatureAnalysis(BaseModel):
    """Schema for technical feature analysis output."""

    key_features: list[str] = Field(description="List of main product features identified")
    technical_strengths: str = Field(description="Technical strengths of the product")
    technical_weaknesses: str = Field(description="Technical limitations or weaknesses")
    innovation_assessment: str = Field(description="Assessment of innovative aspects")
    feature_rating: int = Field(description="Rating from 1-10 of the feature quality", ge=1, le=10)


class SentimentAnalysis(BaseModel):
    """Schema for customer sentiment analysis output."""

    overall_sentiment: str = Field(description="General sentiment toward the product")
    positive_highlights: list[str] = Field(description="Key positive sentiments expressed")
    negative_highlights: list[str] = Field(description="Key negative sentiments expressed")
    emotional_response: str = Field(description="Analysis of emotional customer responses")
    sentiment_rating: int = Field(description="Rating from 1-10 of overall sentiment", ge=1, le=10)


class MarketComparisonAnalysis(BaseModel):
    """Schema for market position and competitor analysis."""

    market_position: str = Field(description="Product's position in the market")
    key_competitors: list[str] = Field(description="Main competing products")
    competitive_advantages: list[str] = Field(description="Advantages over competitors")
    competitive_disadvantages: list[str] = Field(description="Disadvantages compared to competitors")
    value_assessment: str = Field(description="Assessment of price-to-value ratio")
    market_rating: int = Field(description="Rating from 1-10 of market position", ge=1, le=10)


class StructuredOutputSchema(BaseModel):
    """Schema for the final synthesized product review analysis."""

    product_name: str = Field(description="Name of the product being reviewed")
    summary: str = Field(description="Brief executive summary of the product assessment")

    # Feature analysis summary
    key_features_summary: str = Field(description="Summary of key product features")
    technical_assessment: str = Field(description="Assessment of technical aspects")

    # Sentiment analysis summary
    customer_sentiment: str = Field(description="Summary of customer sentiment")
    pros_and_cons: dict[str, list[str]] = Field(description="Organized pros and cons")

    # Market analysis summary
    market_position_summary: str = Field(description="Summary of market position")
    competitive_landscape: str = Field(description="Overview of competitive landscape")

    # Overall assessment
    recommendation: str = Field(description="Final recommendation")
    target_audience: str = Field(description="Ideal customer profile")
    overall_rating: int = Field(description="Overall product rating from 1-10", ge=1, le=10)
    rating_category: RatingLevel = Field(description="Categorical rating assessment")
