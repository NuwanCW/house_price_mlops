from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    val: List = Query(None)


class PredictPayload(BaseModel):
    records: List[Text]

    @validator("records")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "records": [
                    {"val": [[11250, 7, 6, 2.0, 486, 920, 920, 866, 1786, 2]]},
                    {"val": [[11250, 7, 6, 2.0, 486, 920, 920, 866, 1786, 2]]},
                ]
            }
        }
