import pandas as pd
import xgboost as xgb
import joblib
import logging
import numpy as np
import threading # Import thread lock
from .config import settings
from .preprocessing import prepare_data_for_prediction # Keep this

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for loaded objects and a lock ---
model = None
label_encoder = None
scaler = None
model_lock = threading.Lock() # Lock to protect access during reload

# --- Function to load/reload objects ---
def load_model_and_preprocessors():
    """Loads model and preprocessors into global variables. Returns tuple (model, encoder, scaler) or (None, None, None) on failure."""
    logger.info("Attempting to load model and preprocessors...")
    loaded_model, loaded_encoder, loaded_scaler = None, None, None
    try:
        if settings.MODEL_PATH.exists():
            loaded_model = joblib.load(settings.MODEL_PATH)
            logger.info(f"Loaded model from {settings.MODEL_PATH}")
        else:
            logger.error(f"Model file not found: {settings.MODEL_PATH}")

        if settings.LABEL_ENCODER_PATH.exists():
            loaded_encoder = joblib.load(settings.LABEL_ENCODER_PATH)
            logger.info(f"Loaded label encoder from {settings.LABEL_ENCODER_PATH}")
        else:
             logger.error(f"LabelEncoder file not found: {settings.LABEL_ENCODER_PATH}")

        if settings.SCALER_PATH.exists():
            loaded_scaler = joblib.load(settings.SCALER_PATH)
            logger.info(f"Loaded scaler from {settings.SCALER_PATH}")
        else:
            logger.error(f"Scaler file not found: {settings.SCALER_PATH}")

        if loaded_model and loaded_encoder and loaded_scaler:
            logger.info("Model and preprocessors loaded successfully.")
            return loaded_model, loaded_encoder, loaded_scaler
        else:
            logger.error("Failed to load one or more required files (model/encoder/scaler).")
            return None, None, None

    except FileNotFoundError as e: # Specific catch
        logger.error(f"Error loading model/preprocessor file: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {e}", exc_info=True)
        return None, None, None

# --- Function called by training task to update globals ---
def reload_model_and_preprocessors():
    """Reloads the model and preprocessors after training, protected by a lock."""
    global model, label_encoder, scaler
    logger.info("Reloading model and preprocessors...")
    new_model, new_encoder, new_scaler = load_model_and_preprocessors()
    if new_model is not None and new_encoder is not None and new_scaler is not None:
        with model_lock: # Acquire lock before updating globals
            model = new_model
            label_encoder = new_encoder
            scaler = new_scaler # Update scaler reference as well
            logger.info("Global model and preprocessors updated.")
    else:
        logger.error("Reload failed. Model or preprocessors could not be loaded.")


# --- Load initially on startup ---
model, label_encoder, scaler = load_model_and_preprocessors()

# --- Prediction Function ---
def predict_weather(input_df: pd.DataFrame) -> list:
    """Makes weather condition predictions on new data using loaded objects."""
    global model, label_encoder, scaler # Reference globals

    # Check if objects are loaded (could have failed on startup or reload)
    with model_lock: # Acquire lock for reading globals
        current_model = model
        current_encoder = label_encoder
        # Scaler is needed by prepare_data_for_prediction, check its availability implicitly there

    if current_model is None or current_encoder is None:
        logger.error("Model or Label Encoder not available. Cannot predict.")
        return ["Error: Model/Encoder not available"] * len(input_df)

    # Prepare data - this function now loads the scaler and encoder internally
    X_prepared, loaded_encoder_for_decode = prepare_data_for_prediction(input_df)

    if X_prepared is None:
         logger.error("Data preparation for prediction failed.")
         return ["Error: Preprocessing failed"] * len(input_df) # Match input length

    if X_prepared.empty:
        logger.warning("No data left after preprocessing for prediction.")
        return []

    # Check if the encoder needed for decoding was loaded successfully
    if loaded_encoder_for_decode is None:
         logger.error("Label encoder could not be loaded during preprocessing. Cannot decode predictions.")
         try:
             with model_lock: # Lock for prediction
                 predictions_encoded = current_model.predict(X_prepared)
             return [f"Encoded_{p}" for p in predictions_encoded.tolist()]
         except Exception as e:
             logger.error(f"Error during prediction (even without decoding): {e}")
             return ["Error: Prediction failed"] * len(X_prepared)


    logger.info(f"Making predictions on {len(X_prepared)} preprocessed records...")
    try:
        with model_lock: # Acquire lock before using the model
             predictions_encoded = current_model.predict(X_prepared)
        # Inverse transform to get string labels
        predictions_text = loaded_encoder_for_decode.inverse_transform(predictions_encoded)
        logger.info("Predictions generated successfully.")
        return predictions_text.tolist()
    except ValueError as ve:
        if "y contains previously unseen labels" in str(ve):
            logger.error(f"Prediction decoding error: Model predicted labels not seen during encoder training: {ve}")
            # Handle unseen labels - e.g., return 'unknown' or the encoded value
            return [f"Error: Unseen Prediction {p}" for p in predictions_encoded]
        else:
             logger.error(f"ValueError during prediction or inverse transform: {ve}")
             return ["Error: Prediction value error"] * len(X_prepared)
    except Exception as e:
        logger.error(f"Error during prediction or inverse transform: {e}")
        return ["Error: Prediction failed"] * len(X_prepared)