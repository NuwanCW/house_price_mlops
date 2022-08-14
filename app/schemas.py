from typing import List, Literal, Optional, Dict

from fastapi import Query
from pydantic import BaseModel, validator, Field

valid_columns = [
    "LotArea",
    "OverallQual",
    "YearRemodAdd",
    "BsmtQual",
    "BsmtFinSF1",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "GarageCars",
    "YrSold",
]


class Text(BaseModel):
    val: List = Query(None)


class valid_features(BaseModel):
    # val: List = Query(None)
    LotArea: int = Field(..., ge=0, description="Lot area size")
    OverallQual: int = Field(..., ge=1, le=10, description="quality score")
    YearRemodAdd: int = Field(..., ge=1950, description="add removed year")
    # BsmtQual: Optional[Literal['Gd', 'TA', 'Ex', 'Fa']] = None,
    BsmtQual: str = "TA"
    BsmtFinSF1: int = Field(..., ge=0, description="basement f")
    TotalBsmtSF: int = Field(..., ge=0, description="TotalBsmtSF")
    FirststFlrSF: int = Field(..., ge=407, description="1stFlrSF")
    SecondndFlrSF: int = Field(..., ge=0, description="2ndFlrSF")
    GrLivArea: int = Field(..., ge=334, description="GrLivArea")
    GarageCars: int = Field(..., ge=0, description="GarageCars")
    # YrSold: int = Field(...,ge=1950, description='YrSold')


class PredictPayload(BaseModel):
    records: List[List[valid_features]]
    # records: List[Text]

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
