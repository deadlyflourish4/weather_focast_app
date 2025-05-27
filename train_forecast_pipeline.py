import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import calendar
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Import the plotting function from the separate utility file
try:
    from plot_utils import plot_confusion_matrix
except ImportError:
    print("WARN: plot_utils.py not found. Confusion matrix plotting will be disabled.")
    # Define a dummy function if plotting is not essential
    def plot_confusion_matrix(*args, **kwargs):
        print("WARN: Skipping confusion matrix plot.")
        return None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv() # Load .env from project root if present

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR_NAME", "data") # Use env var or default
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR_NAME", "models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Input File paths for the historical dataset
humidity_path = DATA_DIR / "humidity.csv"
pressure_path = DATA_DIR / "pressure.csv"
temperature_path = DATA_DIR / "temperature.csv"
weather_desc_path = DATA_DIR / "weather_description.csv"
wind_dir_path = DATA_DIR / "wind_direction.csv"
wind_speed_path = DATA_DIR / "wind_speed.csv"
city_attr_path = DATA_DIR / "city_attributes.csv"

# Model/Preprocessor Output Paths for the NEXT DAY forecast model
MODEL_PATH = MODEL_DIR / "xgboost_weather_nextday_forecast.pkl" # Specific name
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder_nextday.pkl"    # Specific name
SCALER_PATH = MODEL_DIR / "scaler_nextday.pkl"               # Specific name
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix_nextday_forecast.png"

# Column Names & Feature Definitions
DATETIME_COL = 'datetime'
CITY_COL = 'City'
LABEL_COL = 'weather_condition'
NUMERICAL_COLS_BASE = [ # Base metrics from CSVs
    'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed'
]
LOCATION_COLS = ['Latitude', 'Longitude'] # From city_attributes

# Define forecast horizon (specifically T+24h) and lag/window features
FORECAST_HORIZON_H = 24 # Predict 24 hours ahead
LAG_STEPS_HOURS = [1, 3, 6, 12, 24, 48] # Example hourly lags (include T-1h)
WINDOW_SIZES_HOURS = [6, 12, 24, 48]   # Example hourly windows

# --- Helper Functions ---

def load_and_merge_data():
    """Loads individual historical CSVs, melts, merges them, and adds city attributes."""
    logger.info("Loading data files...")
    try:
        # Load dataframes
        dfs = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in
               ['humidity', 'pressure', 'temperature', 'weather_description',
                'wind_direction', 'wind_speed']}
        df_att = pd.read_csv(city_attr_path)
        dfs[LABEL_COL] = dfs.pop('weather_description') # Rename key
        logger.info("CSV files loaded.")
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}. Ensure CSVs are in '{DATA_DIR.name}'.")
        return None
    except Exception as e:
        logger.error(f"Error loading CSVs: {e}")
        return None

    logger.info("Melting and merging weather data...")
    all_melted = []

    # Determine common cities
    common_cities = None
    for name, df in dfs.items():
         if DATETIME_COL not in df.columns: logger.error(f"'{DATETIME_COL}' missing in {name}.csv"); return None
         df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors='coerce')
         df.dropna(subset=[DATETIME_COL], inplace=True) # Drop rows where datetime conversion failed
         cities = set(df.columns) - {DATETIME_COL}
         if common_cities is None: common_cities = cities
         else: common_cities.intersection_update(cities)

    if not common_cities: logger.error("No common cities found."); return None
    common_cities_list = list(common_cities); logger.info(f"Found {len(common_cities)} common cities.")

    # Melt data
    for name, df in dfs.items():
        try:
            cols_to_melt = [DATETIME_COL] + common_cities_list
            df_filtered = df[cols_to_melt].copy() # Work on a copy
            melted = df_filtered.melt(id_vars=[DATETIME_COL], var_name=CITY_COL, value_name=name)
            all_melted.append(melted.set_index([DATETIME_COL, CITY_COL]))
        except Exception as e: logger.error(f"Error processing {name}.csv: {e}"); return None

    if not all_melted: logger.error("No melted data."); return None
    logger.info("Starting merge (using index)...")
    df_merged = all_melted[0]
    for i, df_next in enumerate(all_melted[1:], 1):
        logger.info(f"Merging dataframe {i+1}/{len(all_melted)}...")
        df_merged = df_merged.join(df_next, how='inner') # Inner join ensures complete records
        if df_merged.empty: logger.warning(f"Empty merge at step {i+1}."); break

    df_merged.reset_index(inplace=True)
    if df_merged.empty: logger.error("Empty after merging weather data."); return None

    # Merge attributes
    logger.info("Merging city attributes...")
    try:
        if not all(col in df_att.columns for col in [CITY_COL, 'Latitude', 'Longitude']): logger.error("Missing columns in city_attributes.csv."); return None
        df_att_subset = df_att[[CITY_COL, 'Latitude', 'Longitude']].copy()
        df_final = pd.merge(df_merged, df_att_subset, on=CITY_COL, how='inner')
        if df_final.empty: logger.warning("Empty after merging attributes."); return None
    except Exception as e: logger.error(f"Attribute merge error: {e}"); return None

    logger.info(f"Data loading/merging complete. Shape: {df_final.shape}")
    return df_final

def handle_description(desc: str) -> str:
    """Aggregates detailed weather descriptions."""
    # ... (same logic as before) ...
    if not isinstance(desc, str): return "unknown"
    desc = desc.lower()
    if any(k in desc for k in ['squall', 'thunderstorm']): return 'thunderstorm'
    if any(k in desc for k in ['drizzle', 'rain', "rainy"]): return 'rainy'
    if any(k in desc for k in ['sleet', 'snow', "snowy"]): return 'snowy'
    if any(k in desc for k in ['cloud', "overcast", "cloudy", "clouds"]): return 'cloudy'
    if any(k in desc for k in ['fog', 'mist', 'haze', "smoke", "dust", "foggy"]): return 'foggy'
    if any(k in desc for k in ['clear', 'sun', "sunny"]): return 'sunny'
    return 'unknown'

def create_lag_window_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers lag, window, and time features for T+24h forecast."""
    logger.info(f"Creating time-series features for DF shape {df.shape}...")
    if DATETIME_COL not in df.columns: logger.error("Datetime column missing."); return pd.DataFrame()
    df = df.sort_values(by=[CITY_COL, DATETIME_COL]).copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

    # --- 1. Target Variable (Condition at T+24h) ---
    target_col_name = f'target_{LABEL_COL}_{FORECAST_HORIZON_H}h'
    df[target_col_name] = df.groupby(CITY_COL)[LABEL_COL].shift(-FORECAST_HORIZON_H)
    initial_len = len(df)
    df.dropna(subset=[target_col_name], inplace=True) # Drop rows where target is missing
    logger.info(f"Dropped {initial_len - len(df)} rows lacking T+{FORECAST_HORIZON_H}h target.")
    if df.empty: logger.error("DataFrame empty after target creation."); return pd.DataFrame()

    # --- 2. Lag Features (Numerical + Original Label String) ---
    logger.info(f"Creating lag features for steps: {LAG_STEPS_HOURS} hours...")
    lag_feature_cols_orig = NUMERICAL_COLS_BASE + [LABEL_COL]
    all_lag_cols = []
    for feature in lag_feature_cols_orig:
        for lag in LAG_STEPS_HOURS:
            lag_col_name = f'{feature}_lag_{lag}h'
            df[lag_col_name] = df.groupby(CITY_COL)[feature].shift(lag)
            all_lag_cols.append(lag_col_name)

    # --- 3. Window Features ---
    logger.info(f"Creating window features for sizes: {WINDOW_SIZES_HOURS} hours...")
    all_window_cols = []
    for feature in NUMERICAL_COLS_BASE:
        for window in WINDOW_SIZES_HOURS:
            window_col_name = f'{feature}_win_{window}h_mean'
            df[window_col_name] = df.groupby(CITY_COL)[feature].shift(1).rolling(window=window, min_periods=max(1, int(window*0.5))).mean()
            all_window_cols.append(window_col_name)
            # Add more window features (std, min, max) if desired here

    # --- 4. Time-Based Features ---
    logger.info("Creating time-based features...")
    df['hour'] = df[DATETIME_COL].dt.hour
    df['dayofweek'] = df[DATETIME_COL].dt.dayofweek
    df['dayofyear'] = df[DATETIME_COL].dt.dayofyear
    df['month'] = df[DATETIME_COL].dt.month
    df['year'] = df[DATETIME_COL].dt.year
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear']/365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear']/365.25)
    time_feature_cols = ['hour', 'dayofweek', 'dayofyear', 'month', 'year', 'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos']

    # --- Drop rows with NaNs introduced by lags/windows ---
    cols_to_check_na = all_lag_cols + all_window_cols
    initial_len = len(df)
    df.dropna(subset=cols_to_check_na, inplace=True)
    logger.info(f"Dropped {initial_len - len(df)} rows due to NaNs in lag/window features.")
    if df.empty: logger.error("DataFrame empty after feature NaN drop."); return pd.DataFrame()

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def undersample_data(X_train, y_train, random_state=None):
    """
    Performs random undersampling on the training data (Pandas DataFrame/Series).
    Handles reset indices after train_test_split.
    """
    if random_state:
        np.random.seed(random_state)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train) # Convert if it's a numpy array

    unique_labels_encoded, counts = np.unique(y_train, return_counts=True)
    label_counts = dict(zip(unique_labels_encoded, counts))
    logger.info(f"Original training label distribution (encoded): {label_counts}")

    if not label_counts or min(counts) == 0:
        logger.warning("Cannot undersample with empty data or zero-count minority class.")
        return X_train, y_train

    minority_class_size = min(counts)
    logger.info(f"Undersampling to minority class size: {minority_class_size}")

    final_indices = [] # Use this list to store the selected *reset* indices
    for label_encoded in label_counts:
        # Find the indices *within the current y_train Series* for this label
        label_indices = y_train[y_train == label_encoded].index.tolist() # Get the actual index values

        if len(label_indices) > minority_class_size:
            # Undersample the *indices* from the current Series index
            chosen_indices = np.random.choice(label_indices, size=minority_class_size, replace=False)
            final_indices.extend(chosen_indices)
            logger.debug(f"Undersampling class {label_encoded} from {len(label_indices)} to {minority_class_size}")
        else:
            # Keep all indices for this minority class
            final_indices.extend(label_indices)
            logger.debug(f"Keeping all {len(label_indices)} samples for class {label_encoded}")

    # Shuffle the final selected indices
    np.random.shuffle(final_indices)

    # Slice using .loc for DataFrames or Series as it uses the index labels
    X_train_undersampled = X_train.loc[final_indices]
    y_train_undersampled = y_train.loc[final_indices]

    # Convert y_train_undersampled back to numpy if needed by XGBoost (usually not)
    # y_train_undersampled = y_train_undersampled.to_numpy()

    logger.info(f"Undersampled training label distribution (encoded): {dict(zip(*np.unique(y_train_undersampled, return_counts=True)))}")

    return X_train_undersampled, y_train_undersampled

# --- Main Training Pipeline ---
if __name__ == "__main__":
    logger.info("--- Starting Weather NEXT DAY FORECAST Training Pipeline ---")

    # 1. Load and Merge Base Data
    df_merged = load_and_merge_data()
    if df_merged is None or df_merged.empty: exit()

    # 2. Initial Preprocessing
    logger.info("Initial preprocessing (label aggregation, NaN drop)...")
    if LABEL_COL not in df_merged.columns: logger.error(f"Label column '{LABEL_COL}' missing."); exit()
    df_merged[LABEL_COL] = df_merged[LABEL_COL].apply(handle_description)
    required_cols_initial = NUMERICAL_COLS_BASE + LOCATION_COLS + [LABEL_COL, CITY_COL, DATETIME_COL]
    missing_req_cols = [col for col in required_cols_initial if col not in df_merged.columns]
    if missing_req_cols: logger.error(f"Missing required columns: {missing_req_cols}."); exit()
    df_clean_initial = df_merged.dropna(subset=required_cols_initial).copy()
    df_clean_initial = df_clean_initial[df_clean_initial[LABEL_COL] != 'unknown']
    if df_clean_initial.empty: logger.error("No data after initial clean."); exit()

    # 3. Feature Engineering (Lags, Windows, Time, Target T+24h)
    df_featured = create_lag_window_time_features(df_clean_initial)
    if df_featured.empty: logger.error("No data after feature engineering."); exit()

    # 4. Label Encoding (Fit on ALL relevant string columns: Target + Lags)
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    try:
        target_col_name = f'target_{LABEL_COL}_{FORECAST_HORIZON_H}h'
        all_labels_for_fit = pd.concat(
            [df_featured[target_col_name].astype(str)] + # Target column
            [df_featured[f'{LABEL_COL}_lag_{lag}h'].fillna('unknown_lag').astype(str) for lag in LAG_STEPS_HOURS] # Lag columns
        ).unique()
        logger.info(f"Fitting LabelEncoder on unique labels: {all_labels_for_fit}")
        label_encoder.fit(all_labels_for_fit)
        logger.info(f"LabelEncoder classes: {label_encoder.classes_}")

        # Apply transform to target column
        df_featured[f'target_encoded_{FORECAST_HORIZON_H}h'] = df_featured[target_col_name].apply(
            lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
        )
        # Apply transform to lagged features
        encoded_lag_cols = []
        for lag in LAG_STEPS_HOURS:
            lag_col = f'{LABEL_COL}_lag_{lag}h'
            encoded_col = f'{lag_col}_encoded'
            df_featured[encoded_col] = df_featured[lag_col].apply(
                lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
            )
            encoded_lag_cols.append(encoded_col)
            df_featured.drop(columns=[lag_col], inplace=True) # Drop original string lag

        # Remove rows with unknown targets or lags (encoded as -1)
        initial_len = len(df_featured)
        df_featured = df_featured[df_featured[f'target_encoded_{FORECAST_HORIZON_H}h'] != -1]
        for col in encoded_lag_cols:
            df_featured = df_featured[df_featured[col] != -1]
        logger.info(f"Removed {initial_len - len(df_featured)} rows with unknown encoded labels (lags or target).")

        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        logger.info(f"LabelEncoder saved to {LABEL_ENCODER_PATH}")

    except Exception as e:
        logger.error(f"Failed during Label Encoding: {e}", exc_info=True); exit()

    if df_featured.empty: logger.error("No data left after encoding."); exit()

    # 5. Define Final Features and Target for T+24h Model
    time_feature_cols = ['hour', 'dayofweek', 'dayofyear', 'month', 'year', 'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos']
    window_feature_cols = [f'{feat}_win_{window}h_mean' for feat in NUMERICAL_COLS_BASE for window in WINDOW_SIZES_HOURS]
    lag_numerical_cols = [f'{feat}_lag_{lag}h' for feat in NUMERICAL_COLS_BASE for lag in LAG_STEPS_HOURS]
    lag_encoded_label_cols = [f'{LABEL_COL}_lag_{lag}h_encoded' for lag in LAG_STEPS_HOURS]

    FINAL_FEATURES = NUMERICAL_COLS_BASE + LOCATION_COLS \
                   + lag_numerical_cols \
                   + lag_encoded_label_cols \
                   + window_feature_cols \
                   + time_feature_cols
                   # NO 'forecast_horizon_h' needed for single horizon model

    TARGET = f'target_encoded_{FORECAST_HORIZON_H}h' # Target is the encoded T+24h label

    # Ensure all defined features exist
    missing_final_features = [f for f in FINAL_FEATURES if f not in df_featured.columns]
    if missing_final_features: logger.error(f"Missing final feature columns: {missing_final_features}"); exit()
    if TARGET not in df_featured.columns: logger.error(f"Target column '{TARGET}' missing."); exit()

    # Drop final NaNs (should be minimal if handled before)
    initial_len = len(df_featured)
    df_model_ready = df_featured.dropna(subset=FINAL_FEATURES + [TARGET]).copy()
    logger.info(f"Dropped {initial_len - len(df_model_ready)} final rows with NaNs.")
    if df_model_ready.empty: logger.error("No data left for modeling."); exit()

    X = df_model_ready[FINAL_FEATURES]
    y = df_model_ready[TARGET].astype(int) # Ensure target is integer

    logger.info(f"Final data for modeling T+{FORECAST_HORIZON_H}h: Features={X.shape}, Target={y.shape}")
    logger.info(f"Target label distribution (encoded):\n{y.value_counts(normalize=True)}")

    # 6. Train-Test Split
    logger.info("Splitting final data...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e: logger.error(f"Train-test split failed: {e}"); exit()

    # 7. Feature Scaling (Fit ONLY on X_train numerical)
    logger.info("Scaling features...")
    scaler = StandardScaler()
    try:
        numerical_features_in_final = [col for col in FINAL_FEATURES if col in X_train.select_dtypes(include=np.number).columns]
        logger.debug(f"Scaling {len(numerical_features_in_final)} numerical features.")

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_features_in_final] = scaler.fit_transform(X_train[numerical_features_in_final])
        X_test_scaled[numerical_features_in_final] = scaler.transform(X_test[numerical_features_in_final])
        logger.info("Features scaled.")
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"StandardScaler saved to {SCALER_PATH}")
    except Exception as e: logger.error(f"Failed during StandardScaler: {e}"); exit()

    # 8. Undersampling (Apply ONLY to the scaled training data)
    logger.info("Applying undersampling to the training set...")
    X_train_undersampled, y_train_undersampled = undersample_data(
        X_train_scaled, y_train, random_state=42
    )
    logger.info(f"Undersampled training set size: {X_train_undersampled.shape}")

    # 9. Model Training (Use undersampled training data, original scaled test data for eval)
    logger.info("Training XGBoost T+24h FORECASTING model...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=250,
        learning_rate=0.07,
        max_depth=7,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )
    try:
        model.fit(X_train_undersampled, y_train_undersampled, # Train on undersampled
                  eval_set=[(X_test_scaled, y_test)],         # Evaluate on original scaled test set
                  #early_stopping_rounds=30,
                  verbose=False)
    except Exception as e: logger.error(f"XGBoost training error: {e}"); exit()
    logger.info("Model training complete.")

    # 10. Save Model
    logger.info(f"Saving trained T+24h forecasting model to {MODEL_PATH}...")
    try: joblib.dump(model, MODEL_PATH); logger.info("Model saved.")
    except Exception as e: logger.error(f"Failed to save model: {e}"); exit()

    # 11. Evaluation (on original scaled test set)
    logger.info("Evaluating model on the original (imbalanced) test set...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    try:
        target_names = label_encoder.classes_.tolist()
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        logger.info(f"Test Set Accuracy (T+24h Forecast Model): {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        logger.info(f"Test Set Accuracy: {accuracy:.4f}")

    # 12. Confusion Matrix (on original scaled test set)
    logger.info("Plotting and saving confusion matrix...")
    try:
        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_confusion_matrix(y_test_labels, y_pred_labels, classes=label_encoder.classes_, normalize=True, title='Normalized Confusion Matrix (T+24h Forecast)', ax=ax)
        fig.savefig(CONFUSION_MATRIX_PATH)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    except Exception as e:
        logger.error(f"Error plotting/saving confusion matrix: {e}")

    logger.info("--- Weather T+24h Forecasting Training Pipeline Finished ---")