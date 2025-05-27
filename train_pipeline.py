import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Import the plotting function from the separate utility file
from weather_forecast_app.plot_utils import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"        # Expects CSVs here
MODEL_DIR = BASE_DIR / "models"     # Saves models/preprocessors here
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure model directory exists

# File paths for the historical dataset
humidity_path = DATA_DIR / "humidity.csv"
pressure_path = DATA_DIR / "pressure.csv"
temperature_path = DATA_DIR / "temperature.csv"
weather_desc_path = DATA_DIR / "weather_description.csv"
wind_dir_path = DATA_DIR / "wind_direction.csv"
wind_speed_path = DATA_DIR / "wind_speed.csv"
city_attr_path = DATA_DIR / "city_attributes.csv" # Contains Lat/Lon

# Model/Preprocessor Output Paths
MODEL_PATH = MODEL_DIR / "xgboost_weather.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Column Names
DATETIME_COL = 'datetime'
CITY_COL = 'City' # Column name in city_attributes.csv and used after melt
LABEL_COL = 'weather_condition'
NUMERICAL_COLS = [
    'humidity', 'pressure', 'temperature', 'wind_direction',
    'wind_speed', 'Latitude', 'Longitude' # From city_attributes
]
FEATURES = NUMERICAL_COLS # Assuming no other categorical features for now

# --- Helper Functions ---

def load_and_merge_data():
    """Loads individual historical CSVs, melts, merges them, and adds city attributes."""
    logger.info("Loading data files...")
    try:
        df_hum = pd.read_csv(humidity_path)
        df_pre = pd.read_csv(pressure_path)
        df_tem = pd.read_csv(temperature_path)
        df_des = pd.read_csv(weather_desc_path)
        df_wdi = pd.read_csv(wind_dir_path)
        df_wsp = pd.read_csv(wind_speed_path)
        df_att = pd.read_csv(city_attr_path)
        logger.info("CSV files loaded.")
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}. Make sure all CSVs are in the '{DATA_DIR.name}' directory.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV loading: {e}")
        return None


    logger.info("Melting and merging weather data...")
    data_frames = {
        'humidity': df_hum, 'pressure': df_pre, 'temperature': df_tem,
        LABEL_COL: df_des, 'wind_direction': df_wdi, 'wind_speed': df_wsp
    }
    all_melted = []

    # Determine common cities across all datasets before melting
    # This helps handle cases where a city might be missing in one file
    common_cities = None
    for name, df in data_frames.items():
         cities = set(df.columns) - {DATETIME_COL}
         if common_cities is None:
             common_cities = cities
         else:
             common_cities.intersection_update(cities)

    if not common_cities:
        logger.error("No common cities found across all weather data CSV files.")
        return None
    logger.info(f"Found {len(common_cities)} common cities across datasets.")
    common_cities_list = list(common_cities)

    for name, df in data_frames.items():
        try:
            if DATETIME_COL not in df.columns:
                 logger.error(f"'{DATETIME_COL}' column missing in {name}.csv")
                 return None
            df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

            # Melt only the common cities + datetime
            cols_to_melt = [DATETIME_COL] + common_cities_list
            melted = df[cols_to_melt].melt(id_vars=[DATETIME_COL], var_name=CITY_COL, value_name=name)
            all_melted.append(melted)
            logger.debug(f"Melted {name}.csv - shape: {melted.shape}")

        except Exception as e:
             logger.error(f"Error processing {name}.csv: {e}")
             return None

    # Merge all melted dataframes
    logger.info("Starting merge of melted data...")
    if not all_melted:
        logger.error("No data frames were successfully melted.")
        return None

    df_merged = all_melted[0]
    for i, df_next in enumerate(all_melted[1:], 1):
        logger.info(f"Merging dataframe {i+1}/{len(all_melted)}...")
        try:
            # Using outer merge initially to see missing data, but inner might be better if we only want complete records
            df_merged = pd.merge(df_merged, df_next, on=[DATETIME_COL, CITY_COL], how='inner') # Changed to inner
            if df_merged.empty:
                logger.warning(f"Merge resulted in empty DataFrame after step {i+1}. Check for matching datetime/city pairs.")
                # Optional: break or return None if empty merge is critical error
        except pd.errors.MergeError as e:
             logger.error(f"Merge error: {e}. Check for inconsistencies.")
             return None
        except Exception as e:
             logger.error(f"Unexpected merge error: {e}")
             return None
        logger.info(f"Merged shape after step {i+1}: {df_merged.shape}")

    if df_merged.empty:
        logger.error("Merging resulted in an empty DataFrame. No common records found across all files.")
        return None

    logger.info("Merging with city attributes...")
    try:
        # Ensure city attribute columns are correctly named
        if not all(col in df_att.columns for col in [CITY_COL, 'Latitude', 'Longitude']):
            logger.error("city_attributes.csv must contain 'City', 'Latitude', 'Longitude' columns.")
            return None

        # Keep only relevant columns and rename City if needed
        df_att_subset = df_att[[CITY_COL, 'Latitude', 'Longitude']].copy()
        # Optional: Rename city column if it's different in city_attributes.csv
        # df_att_subset.rename(columns={'CityNameInAttributes': CITY_COL}, inplace=True)

        # Merge attributes - Use 'inner' to keep only cities with coordinates
        df_final = pd.merge(df_merged, df_att_subset, on=CITY_COL, how='inner')
        if df_final.empty:
             logger.warning("Merge with city attributes resulted in empty DataFrame. Check city name matching.")
             return None

    except Exception as e:
         logger.error(f"Error merging city attributes: {e}")
         return None

    logger.info(f"Data loading and merging complete. Final shape: {df_final.shape}")
    return df_final

def handle_description(desc: str) -> str:
    """Aggregates detailed weather descriptions into broader categories."""
    if not isinstance(desc, str):
        return "unknown"
    weather_condition_lowered = desc.lower()

    if any(key in weather_condition_lowered for key in ['squall', 'thunderstorm']):
        return 'thunderstorm'
    elif any(key in weather_condition_lowered for key in ['drizzle', 'rain', "rainy"]):
        return 'rainy'
    elif any(key in weather_condition_lowered for key in ['sleet', 'snow', "snowy"]):
        return 'snowy'
    elif any(key in weather_condition_lowered for key in ['cloud', "overcast", "cloudy", "clouds"]):
        return 'cloudy'
    elif any(key in weather_condition_lowered for key in ['fog', 'mist', 'haze', "smoke", "dust", "foggy"]):
        return 'foggy'
    elif any(key in weather_condition_lowered for key in ['clear', 'sun', "sunny"]):
        return 'sunny'
    else:
        logger.debug(f"Unhandled weather description, mapping to 'unknown': {desc}")
        return 'unknown'

# --- Main Training Pipeline ---
if __name__ == "__main__":
    logger.info("--- Starting Historical Weather Training Pipeline ---")

    # 1. Load and Merge Data
    df_raw = load_and_merge_data()
    if df_raw is None or df_raw.empty:
        logger.error("Failed to load or merge data. Exiting.")
        exit()

    # 2. Preprocessing
    logger.info("Preprocessing data...")
    # Apply label aggregation
    if LABEL_COL not in df_raw.columns:
        logger.error(f"Label column '{LABEL_COL}' not found after merging.")
        exit()
    df_raw[LABEL_COL] = df_raw[LABEL_COL].apply(handle_description)

    # Handle missing values (drop rows with NaNs in features or label)
    initial_len = len(df_raw)
    required_cols = FEATURES + [LABEL_COL]
    # Check if all required columns exist before dropping NaNs
    missing_req_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_req_cols:
        logger.error(f"Missing required columns after merge: {missing_req_cols}. Check FEATURES and merging logic.")
        exit()

    df_clean = df_raw.dropna(subset=required_cols).copy() # Use copy
    logger.info(f"Dropped {initial_len - len(df_clean)} rows with NaN values in required columns.")

    # Remove 'unknown' categories
    initial_len = len(df_clean)
    df_clean = df_clean[df_clean[LABEL_COL] != 'unknown']
    logger.info(f"Removed {initial_len - len(df_clean)} rows with 'unknown' weather condition.")

    if df_clean.empty:
        logger.error("No data left after cleaning. Exiting.")
        exit()

    # 3. Feature and Label Separation
    X = df_clean[FEATURES]
    y_raw = df_clean[LABEL_COL]
    logger.info(f"Features shape: {X.shape}, Label shape: {y_raw.shape}")
    logger.info(f"Label distribution before split:\n{y_raw.value_counts()}")


    # 4. Label Encoding
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    try:
        y_encoded = label_encoder.fit_transform(y_raw)
        logger.info(f"LabelEncoder fitted with classes: {label_encoder.classes_}")
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        logger.info(f"LabelEncoder saved to {LABEL_ENCODER_PATH}")
    except Exception as e:
        logger.error(f"Failed to fit/save LabelEncoder: {e}")
        exit()

    # 5. Train-Test Split
    logger.info("Splitting data into train and test sets...")
    try:
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    except ValueError as e:
        logger.error(f"Train-test split failed (potentially too few samples per class for stratification): {e}")
        logger.error(f"Class distribution (encoded):\n{np.bincount(y_encoded)}")
        exit()

    # 6. Feature Scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train) # Fit only on training data
        X_test_scaled = scaler.transform(X_test)       # Transform both train and test
        logger.info("Features scaled.")
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"StandardScaler saved to {SCALER_PATH}")
    except Exception as e:
        logger.error(f"Failed to fit/save/transform StandardScaler: {e}")
        exit()

    # 7. Model Training
    logger.info("Training XGBoost model...")
    # Adjust n_estimators, max_depth, learning_rate as needed based on performance/time
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False, # Set to False for newer XGBoost versions
        n_estimators=200,       # Potentially increase for better performance
        learning_rate=0.1,
        max_depth=8,            # Slightly deeper tree
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )

    try:
        model.fit(X_train_scaled, y_train_encoded,
                  eval_set=[(X_test_scaled, y_test_encoded)],
                  verbose=False)            # Set verbose=True to see training progress
    except Exception as e:
        logger.error(f"Error during XGBoost model training: {e}")
        exit()

    logger.info("Model training complete.")

    # 8. Save Model
    logger.info(f"Saving trained model to {MODEL_PATH}...")
    try:
        joblib.dump(model, MODEL_PATH)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        exit()

    # 9. Evaluation
    logger.info("Evaluating model on test set...")
    y_pred_encoded = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    try:
        target_names = label_encoder.classes_.tolist() # Get class names for the report
        report = classification_report(y_test_encoded, y_pred_encoded, target_names=target_names, zero_division=0)
        logger.info(f"Test Set Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        logger.info(f"Test Set Accuracy: {accuracy:.4f} (Report generation failed)")

    # 10. Confusion Matrix
    logger.info("Plotting confusion matrix...")
    try:
        # Use the inverse_transform method of the fitted encoder
        y_test_labels = label_encoder.inverse_transform(y_test_encoded)
        y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
        plot_confusion_matrix(y_test_labels, y_pred_labels, classes=label_encoder.classes_, normalize=True, title='Normalized Confusion Matrix (Historical Data)')
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")


    logger.info("--- Historical Weather Training Pipeline Finished ---")