import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from eda import df, measurement_cols  # cleaned dataset and column list from eda


def train_linear(df, target, features=None, test_size=0.2, random_state=42):
    # select feature columns; by default use only numeric measurements and drop
    # anything that can't be fed to LinearRegression (strings, categories, etc.)
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c not in [target]]
    else:
        # ensure provided features are numeric and not the target
        bad = [c for c in features if c == target or
               not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"Invalid feature columns (must be numeric and not the target): {bad}")

    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }
    return model, X_test, y_test, preds, metrics




def main():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model on the AirQualityUCI dataset"
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="column name to use as the response variable"
    )
    parser.add_argument(
        "--features", type=str, nargs="+",
        help="list of feature column names (default = all other measurements)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="fraction of data to reserve for testing (default 0.20)"
    )

    args = parser.parse_args()

    if args.target not in measurement_cols:
        raise ValueError(
            f"Target must be one of the measurement columns: {measurement_cols}"
        )

    if args.features is not None:
        for feat in args.features:
            if feat not in measurement_cols or feat == args.target:
                raise ValueError(f"Invalid feature column: {feat}")

    model, X_test, y_test, preds, metrics = train_linear(
        df, args.target, features=args.features, test_size=args.test_size
    )

    print("Model coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

