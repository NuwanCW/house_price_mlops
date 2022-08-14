from typing import Dict, List
import numpy as np


def custom_predict(y_pred: np.ndarray) -> np.ndarray:
    """Custom predict function that defaults
    to an index if conditions are not met.
    Args:
        y_prob (np.ndarray): predicted probabilities
        threshold (float): minimum softmax score to predict majority class
        index (int): label index to use if custom conditions is not met.
    Returns:
        np.ndarray: predicted label indices.
    """
    y_preds = [0.0 if p < 0 else p for p in y_pred]
    return np.array(y_preds)


def predict(texts: List, artifacts: Dict) -> List:
    """Predict tags for given texts.
    Args:
        texts (List): raw input texts to classify.
        artifacts (Dict): artifacts from a run.
    Returns:
        List: predictions for input texts.
    """
    # x = texts
    # ordinal_encode = Path(config.DATA_DIR, "encode_cat.pkl")
    # enc = utilload_ordinal_encoding(ordinal_encode)

    # # artifacts["vectorizer"].transform(texts)
    # predict(x)
    print(texts)
    # for x in texts:
    # print(x[3])
    input_data = texts
    input_data[3] = artifacts["ordinal_enc"].transform([[input_data[3]]])[0][0]
    # x[3] = artifacts["ordinal_enc"].transform([[x[3]]])
    # print(texts)
    y_pred = custom_predict(
        y_pred=artifacts["model"].predict([input_data]),
    )
    # tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_data": texts,
            "predicted_price": y_pred[i],
        }
        for i in range(len(y_pred))
    ]
    return predictions
