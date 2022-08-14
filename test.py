import pandas as pd

df = pd.read_csv("./data/train.csv")
print(
    df[
        [
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
            "SalePrice",
            "YrSold",
        ]
    ].head()
)
