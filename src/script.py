# ==============================================================================
# DMML - Course Project: Workers' Compensation Model Zoo (W1-W6)
# Inspired strictly by TD W1 (EDA), W3 (Regression), W5 (Ensembles) & W6 (Boosting)
# ==============================================================================

import os
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Course Constants (TD Standard) ---
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_preprocess_claims(csv_path: str = "workers_comp.csv") -> pd.DataFrame:
    """
    Load dataset and engineer domain features.
    Inspired by W1 Feature Engineering (add_titanic_features).
    """
    if not os.path.exists(csv_path):
        print(f"File '{csv_path}' not found locally. Generating synthetic data for local test...")
        np.random.seed(RANDOM_STATE)
        n = 5000
        df = pd.DataFrame({
            'Age': np.random.randint(18, 65, n),
            'Gender': np.random.choice(['M', 'F'], n),
            'MaritalStatus': np.random.choice(['S', 'M', 'D'], n),
            'PartTimeFullTime': np.random.choice(['P', 'F'], n),
            'WeeklyPay': np.random.exponential(800, n) + 200,
            'HoursWorkedPerWeek': np.random.randint(20, 50, n),
            'DaysWorkedPerWeek': np.random.randint(3, 6, n),
            'DependentChildren': np.random.randint(0, 4, n),
            'DependentsOther': np.random.randint(0, 2, n),
            'DateTimeOfAccident': pd.date_range('2020-01-01', periods=n, freq='h'),
            'DateReported': pd.date_range('2020-01-02', periods=n, freq='h'),
            'InitialCaseEstimate': np.random.exponential(3000, n) + 500,
        })
        df['UltimateIncurredClaimCost'] = df['InitialCaseEstimate'] * np.random.lognormal(0.2, 0.8, n)
    else:
        df = pd.read_csv(csv_path)
    
    # Datetime parsing
    df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])
    df['DateReported'] = pd.to_datetime(df['DateReported'])
    
    # Feature 1: Delay between accident and reporting date (days)
    df['ReportDelayDays'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.total_seconds() / (24 * 3600)
    df['ReportDelayDays'] = df['ReportDelayDays'].clip(lower=0)
    
    # Feature 2 & 3: Temporal markers
    df['AccidentYear'] = df['DateTimeOfAccident'].dt.year
    df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
    
    return df


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a leak-proof ColumnTransformer.
    Inspired by W3/W4 pipeline standards.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor


def evaluate_regression_models(models: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate regression models against baselines without data leakage.
    Applies prediction clipping on log-transformed predictions to prevent exponential explosions.
    Inspired by W3 evaluate_regression_models().
    """
    results = []
    
    # Log-bounds for safe exponentiation (Clipping)
    max_log_val = np.log1p(y_train).max()
    min_log_val = np.log1p(y_train).min()
    
    for name, model in models.items():
        start_time = perf_counter()
        
        # Fit model on log-transformed target (log1p)
        model.fit(X_train, np.log1p(y_train))
        fit_time = perf_counter() - start_time
        
        # Predict & clip log predictions to safe bounds
        raw_log_preds = model.predict(X_test)
        clipped_log_preds = np.clip(raw_log_preds, a_min=min_log_val, a_max=max_log_val)
        
        # Convert back to real dollars (expm1)
        y_pred = np.expm1(clipped_log_preds)
        
        # Standard TD Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE calculation
        valid = y_test > 0
        mape = np.mean(np.abs((y_test[valid] - y_pred[valid]) / y_test[valid])) * 100
        
        results.append({
            "model": name,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape_pct": round(mape, 2),
            "r2_score": round(r2, 4),
            "fit_time_sec": round(fit_time, 4)
        })
        
    return pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)


# --- Execution Pipeline ---
if __name__ == "__main__":
    data_file = "workers_comp.csv"
    
    raw_df = load_and_preprocess_claims(data_file)
    
    target_col = 'UltimateIncurredClaimCost'
    numeric_features = [
        'Age', 'WeeklyPay', 'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 
        'DependentChildren', 'DependentsOther', 'ReportDelayDays', 'InitialCaseEstimate'
    ]
    categorical_features = ['Gender', 'MaritalStatus', 'PartTimeFullTime']
    
    X = raw_df[numeric_features + categorical_features]
    y = raw_df[target_col]
    
    # Leak-proof train/test split (W3 convention)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Data Split: Train = {X_train.shape} rows | Test = {X_test.shape} rows")
    
    # Preprocessing & Model Zoo
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    models = {
        # --- Baselines (W3) ---
        "dummy_mean_baseline": Pipeline([
            ('prep', preprocessor),
            ('reg', DummyRegressor(strategy='mean'))
        ]),
        "linear_regression": Pipeline([
            ('prep', preprocessor),
            ('reg', LinearRegression())
        ]),
        "ridge_regression": Pipeline([
            ('prep', preprocessor),
            ('reg', Ridge(alpha=10.0))
        ]),
        "lasso_regression": Pipeline([
            ('prep', preprocessor),
            ('reg', Lasso(alpha=1.0))
        ]),
        # --- Tree & Ensemble Models (W5) ---
        "random_forest": Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "gradient_boosting": Pipeline([
            ('prep', preprocessor),
            ('reg', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE))
        ]),
        # --- Advanced Boosting (W6) ---
        "lightgbm": Pipeline([
            ('prep', preprocessor),
            ('reg', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE, verbose=-1))
        ])
    }
    
    benchmark_df = evaluate_regression_models(models, X_train, X_test, y_train, y_test)
    
    print("\n=== COURSE BENCHMARK RESULTS (W1-W6 MODELS) ===")
    print(benchmark_df.to_string(index=False))
