from pydantic import BaseModel, Field


class StructuredOutputSchema(BaseModel):
    sentiment_score: float = Field(
        default=0.0,
        description="A sentiment score ranging from -1.0 to 1.0, where -1.0 indicates extremely negative sentiment, 0.0 is neutral, and 1.0 is extremely positive."
    )