"""
Linear regression to produce correlations and predict future values. 

This script loads the raw data, adds the day and weekday flag columns, creates a new data csv with those columns, and runs a linear regression model. 

Outputs a CSV with added day and weekday flags, correlation values, and test values. 
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from eda import df, measurement_cols


def run_ols_for_pollutants(df: pd.DataFrame, measurement_cols: list[str], output_dir: str) -> pd.DataFrame:
    results = []
    #the main features we are trying to test
    features = ["Day", "Weekday"]

    #loop through each pollutant and run a linear regression with day and weekday as features, then save the results
    for target in measurement_cols:
        if target in ["Date", "Time", "Datetime", "hour", "DayNight", "WeekdayNum", "WeekdayWeekend"]:
            continue
        if not np.issubdtype(df[target].dtype, np.number):
            continue
        df_sub = df[[target, "Day", "Weekday"]].dropna()
        if df_sub.shape[0] < 50:
            continue
        
        #extracting features and running linear regression
        X = df_sub[features]
        y = df_sub[target]
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X)

        #appending results to list
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
    #output results to a CSV file in the output directory
    out.to_csv(os.path.join(output_dir, "linear_reg_day_weekday_results.csv"), index=False)
    return out


def run_predictive_evaluation(df: pd.DataFrame, target: str) -> dict:
    features = ["Day", "Weekday"]
    df_sub = df[[target, "Day", "Weekday"]].dropna()
    X = df_sub[features]
    y = df_sub[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #80-20 train-test split with a fixed random state for reproducibility

    lr = LinearRegression()
    lr.fit(X_train, y_train) #run linear regression on the training set
    y_pred = lr.predict(X_test)
    
    #evaluate the predictions using RMSE, R-squared, and MAE, and return the results in a dictionary that's saved to a CSV file
    return {
        "target": target,
        "n": int(df_sub.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
    }


def evaluate_all_pollutants(df: pd.DataFrame, measurement_cols: list[str], output_dir: str) -> pd.DataFrame:
    metrics = []
    #loop through each pollutant and run the predictive evaluation function
    for target in measurement_cols:
        if target in ["Date", "Time", "Datetime", "hour", "DayNight", "WeekdayNum", "WeekdayWeekend"]:
            continue
        if not np.issubdtype(df[target].dtype, np.number):
            continue
        df_sub = df[[target, "Day", "Weekday"]].dropna()
        if df_sub.shape[0] < 50:
            continue
        m = run_predictive_evaluation(df, target)
        metrics.append(m)

    out = pd.DataFrame(metrics)
    out = out.sort_values(by="r2", ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "linear_regression_test_metrics_by_pollutant.csv"), index=False) #output is saved to a CSV file in the output directory
    return out


def day_week_correlation(df: pd.DataFrame, measurement_cols: list[str], output_dir: str) -> pd.DataFrame:
    numeric_cols = [c for c in measurement_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    corr = df[numeric_cols + ["Day", "Weekday"]].corr()
    corr_details = pd.DataFrame({
        "target": numeric_cols,
        "corr_with_day": corr.loc[numeric_cols, "Day"].values,
        "corr_with_weekday": corr.loc[numeric_cols, "Weekday"].values,
    }) #create a DataFrame with the correlation details for each numeric pollutant column with the Day and Weekday indicators
    os.makedirs(output_dir, exist_ok=True)
    corr_details.to_csv(os.path.join(output_dir, "day_weekday_correlations.csv"), index=False) #output results into a CSV file in the output directory
    return corr_details


def main():
    global df, measurement_cols

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    output_dir = os.path.abspath(output_dir)
    data_dir = os.path.join(script_dir, "..", "data")
    data_dir = os.path.abspath(data_dir)

    #run OLS regression for each pollutant with Day and Weekday as features, and save the results to a CSV file in the output directory
    ols_results = run_ols_for_pollutants(df, measurement_cols, output_dir)
    print(f"Linear regression summary saved to {os.path.join(output_dir, 'linear_reg_day_weekday_results.csv')}")

    #print the top 20 results to the console for quick review
    print("\nDay/Night and Weekday/Weekend regression coefficients by pollutant:")
    display_cols = ["target", "n", "r_squared", "intercept", "coef_day", "coef_weekday"]
    print(ols_results[display_cols].head(20).to_string(index=False))

    #evaluate the predictive performance of the linear regression model for each pollutant using an 80-20 train-test split, and save the results to a CSV file in the output directory
    test_metrics = evaluate_all_pollutants(df, measurement_cols, output_dir)
    print("\nTest-split accuracy/eval metrics for each pollutant (saved in output/linear_regression_test_metrics_by_pollutant.csv):")
    print(test_metrics.to_string(index=False))

    #evaluate correlation of each pollutant with the Day and Weekday indicators, and save the results to a CSV file in the output directory
    corr_details = day_week_correlation(df, measurement_cols, output_dir)
    print("\nCorrelation with Day/Night and Weekday/Weekend binary indicators (saved in output/day_weekday_correlations.csv):")
    print(corr_details.to_string(index=False))

    #save day/week flags to data CSV
    out_cols = [c for c in ["Date", "Time", "Datetime", "Day", "Weekday"] if c in df.columns]
    out_cols += [c for c in measurement_cols if c in df.columns and c not in out_cols]
    os.makedirs(data_dir, exist_ok=True)
    dayweek_file = os.path.join(data_dir, "day_weekday_data_flags.csv")
    df[out_cols].to_csv(dayweek_file, index=False)
    print(f"\nSaved day/week flag dataset to {dayweek_file} (1=true, 0=false for Day and Weekday)")

if __name__ == "__main__":
    main()
