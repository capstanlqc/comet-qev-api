from fastapi import HTTPException

from evaluate import load
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

class InputData(BaseModel):
    source: List[str]
    hypothesis: List[str]
    reference: List[str]

comet_metric = load('comet')

app = FastAPI()


@app.post("/evaluate", response_model=List[float])
def process_items(data: InputData):

    try:
        assert len(data.source) == len(data.hypothesis) == len(data.reference)
    except AssertionError:
        raise HTTPException(status_code=400, detail="The three groups (source, hypothesis and reference) must have the same number of segments.")

    results = comet_metric.compute(
        predictions=data.hypothesis,
        references=data.reference,
        sources=data.source
    )
    return [round(v, 3) for v in results["scores"]]