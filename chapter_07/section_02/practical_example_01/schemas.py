""" 
Schema definitions for the Document Analysis System.
These schemas define the structure for the various analysis outputs.
"""

# Third Party
from pydantic import BaseModel, Field


class MethodologyAnalysis(BaseModel):
    """Schema for methodology analysis output."""
    approach_validity: str = Field(description="Assessment of the research approach validity")
    methodology_strengths: str = Field(description="Key strengths of the methodology")
    methodology_limitations: str = Field(description="Limitations or weaknesses in the methodology")
    overall_rating: int = Field(description="Rating from 1-10 of the methodology quality")


class ResultsAnalysis(BaseModel):
    """Schema for results analysis output."""
    data_interpretation: str = Field(description="Assessment of data interpretation quality")
    findings_validity: str = Field(description="Validity of the reported findings")
    statistical_rigor: str = Field(description="Evaluation of statistical methods used")
    overall_rating: int = Field(description="Rating from 1-10 of the results quality")


class ImplicationsAnalysis(BaseModel):
    """Schema for implications analysis output."""
    practical_applications: str = Field(description="Potential practical applications")
    broader_impact: str = Field(description="Assessment of broader impact on the field")
    future_directions: str = Field(description="Suggested future research directions")
    overall_rating: int = Field(description="Rating from 1-10 of the implications significance")


class SynthesisAnalysis(BaseModel):
    """Schema for the final synthesis analysis."""
    methodology_summary: str = Field(description="Summary of methodology analysis")
    results_summary: str = Field(description="Summary of results analysis")
    implications_summary: str = Field(description="Summary of implications analysis")
    overall_assessment: str = Field(description="Comprehensive assessment of the document")
    overall_rating: int = Field(description="Overall document quality rating from 1-10")
