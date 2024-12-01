from pydantic import BaseModel, Field
from enum import Enum


class MusicGenre(str, Enum):
    classical = "classical"
    rock = "rock"


class StructuredOutputSchema(BaseModel):
    music_genre_key: MusicGenre = Field(
        description="The identified genre of the music, either 'classical' or 'rock'."
    )
