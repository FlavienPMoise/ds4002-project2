import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from eda import df, measurement_cols



def run_ols_for_pollutants(df: pd.DataFrame, measurement_cols: list[str], output_dir: str) -> pd.DataFrame:
    results = []
    features = ["Day", "Weekday"]

    for target in measurement_cols:
        if target in ["Date", "Time", "Datetime", "hour", "DayNight", "WeekdayNum", "WeekdayWeekend"]:
            continue
        # ignore non-numeric columns or those with insufficient data
        if not np.issubdtype(df[target].dtype, np.number):
            continue
        df_sub = df[[target, "Day", "Weekday"]].dropna()
        if df_sub.shape[0] < 50:
            continue

        X = df_sub[features]
        y = df_sub[target]
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X)

        results.append({
            "target": target,
            "n": int(df_sub.shape[0]),
            "r_squared": float(r2_score(y, y_pred)),
            "intercept": float(model.intercept_),
            "coef_day": float(model.coef_[0]),
            "coef_weekday": float(model.coef_[1]),
        })

    out = pd.DataFrame(results)
    out = out.sort_values(by="r_squared", ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "linear_reg_day_weekday_results.csv"), index=False)
    return out


def run_predictive_evaluation(df: pd.DataFrame, target: str, output_dir: str) -> dict:
    features = ["Day", "Weekday"]
    df_sub = df[[target, "Day", "Weekday"]].dropna()
    X = df_sub[features]
    y = df_sub[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    metrics = {
        "target": target,
        "coef_day": float(lr.coef_[0]),
        "coef_weekday": float(lr.coef_[1]),
        "intercept": float(lr.intercept_),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    return metrics


def main():
    global df, measurement_cols

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Save outputs relative to repository root (parent of scripts)
    output_dir = os.path.join(script_dir, "..", "output")
    output_dir = os.path.abspath(output_dir)
    print(f"Using output directory: {output_dir}")

    ols_results = run_ols_for_pollutants(df, measurement_cols, output_dir)
    summary_path = os.path.join(output_dir, "linear_reg_day_weekday_results.csv")
    print(f"Linear regression summary saved to {summary_path}")

    print("\nDay/Night and Weekday/Weekend regression coefficients by pollutant:")
    display_cols = ["target", "n", "r_squared", "intercept", "coef_day", "coef_weekday"]
    print(ols_results[display_cols].head(15).to_string(index=False))

    top_targets = ols_results.sort_values(by="r_squared", ascending=False).head(3)["target"].tolist()
    eval_metrics = []
    for t in top_targets:
        m = run_predictive_evaluation(df, t, output_dir)
        eval_metrics.append(m)

    eval_df = pd.DataFrame(eval_metrics)
    eval_df.to_csv(os.path.join(output_dir, "linear_regression_predictive_evaluation.csv"), index=False)
    print("\nPredictive evaluation saved to output/linear_regression_predictive_evaluation.csv")
    print(eval_df.to_string(index=False))

if __name__ == "__main__":
    main()
