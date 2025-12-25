from pydantic import BaseModel
from typing import Optional

class IndexRequest(BaseModel):
    type: Optional[str] = "all"  # captions | stories | all

class SearchResponseItem(BaseModel):
    id: str
    score: float
    payload: dict

class SearchResponse(BaseModel):
    results: list[SearchResponseItem]
