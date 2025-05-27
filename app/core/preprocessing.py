import pandas as pd
import joblib
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def preprocess_and_split(df: pd.DataFrame, fit_scalers_encoders: bool = True):
    """
    Cleans, encodes labels, scales features, and splits data.
    Saves fitted objects if fit_scalers_encoders=True.
    Returns: X_train, X_test, y_train, y_test, label_encoder (or None if errors)
    """
    logger.info(f"Starting preprocessing and splitting on DataFrame with {len(df)} rows.")
    label_encoder = None
    scaler = None

    # 1. Aggregate Labels First
    if settings.LABEL_COL not in df.columns:
        logger.error(f"Label column '{settings.LABEL_COL}' not found in DataFrame.")
        return None, None, None, None, None
    logger.info("Aggregating weather condition labels...")
    df[settings.LABEL_COL] = df[settings.LABEL_COL].apply(handle_description)

    # 2. Handle Missing Values (Drop rows with NaNs in features or non-unknown label)
    initial_len = len(df)
    required_cols = settings.FEATURES + [settings.LABEL_COL]
    df.dropna(subset=required_cols, inplace=True)
    logger.info(f"Dropped {initial_len - len(df)} rows with NaN values in required columns.")

    # 3. Remove 'unknown' categories
    initial_len = len(df)
    df = df[df[settings.LABEL_COL] != 'unknown']
    logger.info(f"Removed {initial_len - len(df)} rows with 'unknown' weather condition.")

    if df.empty:
        logger.error("DataFrame is empty after cleaning NaNs and unknowns.")
        return None, None, None, None, None

    # 4. Feature and Label Separation
    try:
        X = df[settings.FEATURES].copy()
        y_raw = df[settings.LABEL_COL].copy()
    except KeyError as e:
        logger.error(f"Feature/Label column missing after cleaning: {e}")
        return None, None, None, None, None

    logger.info(f"Cleaned data shape: Features={X.shape}, Labels={y_raw.shape}")

    # 5. Label Encoding
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    try:
        y_encoded = label_encoder.fit_transform(y_raw)
        logger.info(f"LabelEncoder fitted with classes: {label_encoder.classes_}")
        if fit_scalers_encoders:
            joblib.dump(label_encoder, settings.LABEL_ENCODER_PATH)
            logger.info(f"LabelEncoder saved to {settings.LABEL_ENCODER_PATH}")
    except Exception as e:
        logger.error(f"Failed to fit/save LabelEncoder: {e}")
        return None, None, None, None, None


    # 6. Train-Test Split (Stratified)
    logger.info("Splitting data into train and test sets...")
    try:
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    except ValueError as e:
        logger.error(f"Train-test split failed (potentially too few samples per class): {e}")
        # Log class distribution
        logger.error(f"Class distribution:\n{y_raw.value_counts()}")
        return None, None, None, None, None


    # 7. Feature Scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train) # Fit only on training data
        X_test_scaled = scaler.transform(X_test)       # Transform both train and test
        logger.info("Features scaled.")
        if fit_scalers_encoders:
            joblib.dump(scaler, settings.SCALER_PATH)
            logger.info(f"StandardScaler saved to {settings.SCALER_PATH}")
    except Exception as e:
        logger.error(f"Failed to fit/save/transform StandardScaler: {e}")
        return None, None, None, None, None

    logger.info("Preprocessing and splitting finished successfully.")
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, label_encoder


def prepare_data_for_prediction(df: pd.DataFrame):
    """
    Loads fitted preprocessors and prepares new data for prediction.
    Returns preprocessed features (X) and the loaded label encoder.
    Handles potential NaNs by dropping rows.
    """
    logger.info(f"Preparing {len(df)} records for prediction...")
    label_encoder = None
    scaler = None

    # 1. Select Features (ensure order matches training)
    try:
        X = df[settings.FEATURES].copy()
    except KeyError as e:
        logger.error(f"Input data missing required feature columns for prediction: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error selecting features for prediction: {e}")
        return None, None

    # 2. Handle NaNs in features before scaling
    initial_len = len(X)
    X.dropna(subset=settings.NUMERICAL_COLS, inplace=True)
    if len(X) < initial_len:
        logger.warning(f"Dropped {initial_len - len(X)} rows with NaN in numerical features during prediction prep.")

    if X.empty:
        logger.error("DataFrame is empty after dropping NaNs during prediction prep.")
        return None, None


    # 3. Load and Apply Scaler
    try:
        scaler = joblib.load(settings.SCALER_PATH)
        X[settings.NUMERICAL_COLS] = scaler.transform(X[settings.NUMERICAL_COLS])
        logger.info("Applied loaded StandardScaler for prediction.")
    except FileNotFoundError:
        logger.error(f"StandardScaler file not found at {settings.SCALER_PATH}. Cannot scale features.")
        return None, None
    except ValueError as e:
         logger.error(f"Scaler transform error (likely column mismatch): {e}")
         return None, None
    except Exception as e:
        logger.error(f"Error loading/applying StandardScaler for prediction: {e}")
        return None, None

    # 4. Load Label Encoder (needed for inverse transforming predictions later)
    try:
        label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)
        logger.info(f"Loaded LabelEncoder for prediction decoding.")
    except FileNotFoundError:
        logger.error(f"LabelEncoder file not found at {settings.LABEL_ENCODER_PATH}. Cannot decode predictions.")
        # Can still return X, but decoding will fail later
        return X, None
    except Exception as e:
        logger.error(f"Error loading LabelEncoder for prediction: {e}")
        return X, None

    logger.info(f"Prediction data preparation finished. Shape: {X.shape}")
    return X, label_encoder