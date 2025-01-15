from pydantic import BaseModel
from typing import Dict

class Ingredient(BaseModel):
    name: str
    grams: float

class NutritionData(BaseModel):
    ingredients: Dict[str, Dict[str, float]]
    total_meal: Dict[str, float]
