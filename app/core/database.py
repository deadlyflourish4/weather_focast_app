import pandas as pd
from pymongo import MongoClient, UpdateOne, errors
import logging
import numpy as np
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mongo_client = None
db = None
collection = None

def connect_to_mongo():
    """Establishes connection to MongoDB."""
    global mongo_client, db, collection
    if mongo_client is None:
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGO_URI}...")
            mongo_client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command('ismaster') # Check connection
            db = mongo_client[settings.MONGO_DB_NAME]
            collection = db[settings.MONGO_COLLECTION_NAME]
            # Create index for efficient upsert/querying (datetime, province)
            collection.create_index([("datetime", 1), ("province", 1)], unique=True, background=True)
            logger.info("MongoDB connection successful.")
        except errors.ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            mongo_client = None
            db = None
            collection = None
        except Exception as e:
             logger.error(f"An unexpected error occurred during MongoDB connection: {e}")
             mongo_client = None
             db = None
             collection = None

def close_mongo_connection():
    """Closes MongoDB connection."""
    global mongo_client
    if mongo_client:
        logger.info("Closing MongoDB connection.")
        mongo_client.close()
        mongo_client = None
        db = None
        collection = None

def save_data_to_mongo(df: pd.DataFrame):
    """Saves DataFrame data to MongoDB, performing upserts based on datetime and province."""
    connect_to_mongo() # Ensure connection is active
    if collection is None:
        logger.error("MongoDB connection not established. Cannot save data.")
        return 0

    if df.empty:
        logger.info("Received empty DataFrame, nothing to save to MongoDB.")
        return 0

    # Convert datetime to Python native datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
       try:
           df['datetime'] = pd.to_datetime(df['datetime'])
       except Exception as e:
           logger.error(f"Error converting datetime column: {e}")
           return 0 # Cannot proceed without valid datetime

    # Replace NaNs with None for MongoDB compatibility BEFORE converting to dict
    # Be specific about columns where NaNs are expected/allowed
    cols_to_clean = settings.NUMERICAL_COLS + [settings.LABEL_COL]
    for col in cols_to_clean:
        if col in df.columns:
             # Replace Pandas NA and Numpy NaN with None
             df[col] = df[col].replace({pd.NA: None, np.nan: None})
             # Attempt numeric conversion again if needed, replace errors with None
             if col in settings.NUMERICAL_COLS:
                 df[col] = pd.to_numeric(df[col], errors='coerce').replace({np.nan: None})


    logger.info(f"Preparing {len(df)} records for MongoDB upsert...")
    try:
        records = df.to_dict('records')
    except Exception as e:
        logger.error(f"Error converting DataFrame to records (check data types/NaNs): {e}")
        return 0

    operations = []
    for record in records:
        # Ensure essential fields for the filter are present
        if record.get('datetime') is None or record.get('province') is None:
            logger.warning(f"Skipping record due to missing datetime or province: {record}")
            continue

        filter_doc = {
            'datetime': record['datetime'],
            'province': record['province']
        }
        # Prepare update doc, ensuring no complex numpy types remain if missed above
        update_doc_set = {}
        for k, v in record.items():
            if isinstance(v, (np.int64, np.int32)): update_doc_set[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)): update_doc_set[k] = float(v)
            elif pd.isna(v): update_doc_set[k] = None # Ensure NaN becomes None
            else: update_doc_set[k] = v

        update = {"$set": update_doc_set}
        operations.append(UpdateOne(filter_doc, update, upsert=True))

    if not operations:
        logger.info("No valid operations generated for MongoDB.")
        return 0

    saved_count = 0
    try:
        logger.info(f"Executing {len(operations)} bulk write operations...")
        result = collection.bulk_write(operations, ordered=False) # ordered=False might be faster but harder to debug errors
        saved_count = (result.upserted_count or 0) + (result.modified_count or 0)
        logger.info(f"MongoDB bulk write complete. Upserted: {result.upserted_count}, Matched: {result.matched_count}, Modified: {result.modified_count}")
    except errors.BulkWriteError as bwe:
        logger.error(f"MongoDB bulk write error details: {bwe.details}")
        # Count successes even if some failed (optional)
        saved_count = (bwe.details.get('nUpserted', 0) or 0) + (bwe.details.get('nModified', 0) or 0)
        logger.error(f"Partial success count (upserted/modified): {saved_count}")
    except Exception as e:
        logger.error(f"Unexpected error saving data to MongoDB: {e}")
        saved_count = 0

    return saved_count

def load_data_from_mongo() -> pd.DataFrame:
    """Loads all data from the MongoDB collection into a DataFrame."""
    connect_to_mongo() # Ensure connection is active
    if collection is None:
        logger.error("MongoDB connection not established. Cannot load data.")
        return pd.DataFrame()

    try:
        logger.info(f"Loading data from MongoDB collection: {settings.MONGO_COLLECTION_NAME}...")
        cursor = collection.find({}, {"_id": 0}) # Exclude the MongoDB _id field
        df = pd.DataFrame(list(cursor))
        if df.empty:
            logger.warning("MongoDB collection is empty.")
        else:
            # Attempt datetime conversion, handle potential errors
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception as e:
                logger.error(f"Error converting 'datetime' column to datetime objects: {e}. Check data consistency in MongoDB.")
                # Decide how to handle: drop rows, return empty, or try to continue?
                # For now, log error and return potentially problematic DataFrame
            logger.info(f"Loaded {len(df)} records from MongoDB.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from MongoDB: {e}")
        return pd.DataFrame()