import os
from dotenv import load_dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Determine the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f".env file loaded from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}. Using default settings or environment variables.")


class Settings:
    WORLD_WEATHER_API_KEY: str = os.getenv("WORLD_WEATHER_API_KEY", "YOUR_DEFAULT_KEY")
    PROVINCES_DATA_PATH: Path = BASE_DIR / os.getenv("PROVINCES_DATA_PATH", "data/provinces_vn.json")
    MODEL_PATH: Path = BASE_DIR / os.getenv("MODEL_PATH", "models/xgboost_weather.pkl")
    LABEL_ENCODER_PATH: Path = BASE_DIR / os.getenv("LABEL_ENCODER_PATH", "models/label_encoder.pkl")
    SCALER_PATH: Path = BASE_DIR / os.getenv("SCALER_PATH", "models/scaler.pkl")

    # --- MongoDB Settings ---
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "weather_db")
    MONGO_COLLECTION_NAME: str = os.getenv("MONGO_COLLECTION_NAME", "hourly_data")

    # --- Scheduler Settings ---
    DAILY_FETCH_HOUR: int = int(os.getenv("DAILY_FETCH_HOUR", 1)) # Default 1 AM
    DAILY_RETRAIN_HOUR: int = int(os.getenv("DAILY_RETRAIN_HOUR", 2)) # Default 2 AM

    # --- Feature/Label Settings ---
    NUMERICAL_COLS = [
        'humidity', 'pressure', 'temperature', 'wind_direction',
        'wind_speed', 'latitude', 'longitude'
    ]
    NOMINAL_COLS = []
    LABEL_COL = 'weather_condition'
    FEATURES = NUMERICAL_COLS + NOMINAL_COLS

settings = Settings()

# Ensure directories exist
try:
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.PROVINCES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checked/created directories: {settings.MODEL_PATH.parent}, {settings.PROVINCES_DATA_PATH.parent}")
except Exception as e:
    logger.error(f"Error creating necessary directories: {e}")

# Log critical settings (mask API key)
logger.info(f"WorldWeatherOnline API Key: {'*' * (len(settings.WORLD_WEATHER_API_KEY) - 4)}{settings.WORLD_WEATHER_API_KEY[-4:]}" if settings.WORLD_WEATHER_API_KEY else "Not Set")
logger.info(f"MongoDB URI: {settings.MONGO_URI.replace(settings.MONGO_URI.split('@')[-1], '****') if '@' in settings.MONGO_URI else settings.MONGO_URI}") # Basic masking
logger.info(f"Province Data Path: {settings.PROVINCES_DATA_PATH}")
logger.info(f"Model Path: {settings.MODEL_PATH}")