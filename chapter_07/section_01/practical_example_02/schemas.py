""" """

# Standard Library 
from enum import Enum 
 
# Third Party 
from pydantic import BaseModel, Field 
  

class PerformanceLevel(str, Enum): 
    excellent = "Excellent" 
    satisfactory = "Satisfactory"  
    needs_improvement = "Needs Improvement" 
 

class OutputSchema(BaseModel): 
    score: int = Field(description="Numerical assessment score from 0-100") 
    performance: PerformanceLevel = Field(description="Overall performance classification") 
    feedback: str = Field(description="Detailed evaluative feedback on the assessment") 
    passed: bool = Field(description="Whether the assessment meets passing requirements") 
