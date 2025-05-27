from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Query, Request, Depends
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from fastapi.responses import JSONResponse # To customize error response
from datetime import datetime, timedelta
import pandas as pd
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import os
import uuid # For task IDs
import threading # For locking task dictionary
from typing import Dict, Any, Optional
# from .core.utils import merge_today_data_into_csv

# Import core components
try:
    from .core import config, prediction, data_loader, training, database, current_weather
    from .models import PredictRequest, PredictResponse, TrainResponse, BackgroundTaskResponse, XGBoostTrainParams, CurrentWeatherResponse, RetrainStatus, RetrainTriggerResponse
    from pymongo import errors
    TRAINING_ENABLED = True
except ImportError as e:
    logging.exception(f"Could not import training/data related modules: {e}")
    from .core import config, prediction
    from .models import PredictRequest, PredictResponse
    TRAINING_ENABLED = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vietnam Weather Forecast API",
    version="1.0.1",
    description="API for predicting weather conditions and managing data/retraining."
)

retrain_tasks: Dict[str, RetrainStatus] = {}
tasks_lock = threading.Lock() # Protect concurrent access

# if TRAINING_ENABLED and training.CM_PLOT_DIR.exists():
#     app.mount("/static", StaticFiles(directory=str(training.BASE_DIR / "static")), name="static")
#     logger.info(f"Mounted static directory at /static serving from {training.BASE_DIR / 'static'}")
# else:
#      logger.warning("Static file serving for CM plots not configured (Training disabled or dir missing).")

training_in_progress = False

# Scheduler setup
scheduler = BackgroundScheduler(daemon=True) if TRAINING_ENABLED else None

def daily_fetch_job():
    if not TRAINING_ENABLED:
        return
    logger.info("Running daily data fetch job...")
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
        start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Fetching data for date range: {start_date} to {end_date}")
        saved_count = data_loader.fetch_and_store_data_for_provinces(start_date, end_date)
        logger.info(f"Daily fetch job completed. Saved {saved_count} new/updated records.")

        from .core.utils import merge_today_data_into_csv
        from .core.training import incremental_train_model
        merge_today_data_into_csv(start_date)
        incremental_train_model(start_date)
    except Exception as e:
        logger.error(f"Error in daily fetch job: {e}", exc_info=True)

def daily_retrain_job():
    if not TRAINING_ENABLED:
        return
    logger.info("Triggering daily retraining job...")
    try:
        _, accuracy = training.train_model()
        if accuracy is not None:
            logger.info(f"Scheduled retraining completed. Test Accuracy: {accuracy:.4f}")
        else:
            logger.error("Scheduled retraining failed.")
    except Exception as e:
        logger.error(f"Error in daily retrain job: {e}", exc_info=True)

def daily_retrain_job_wrapper():
    """Wrapper for scheduled retraining using default parameters."""
    global training_in_progress # Declare intent to modify global
    if not TRAINING_ENABLED: return
    if training_in_progress:
        logger.warning("Scheduled retraining skipped: a training process is already running.")
        return
    logger.info("Triggering daily retraining job with default parameters...")
    training_in_progress = True
    try:
        _, accuracy, _ = training.train_model() # Uses defaults
        if accuracy is not None:
            logger.info(f"Scheduled retraining completed. Test Accuracy: {accuracy:.4f}")
        else:
            logger.error("Scheduled retraining failed.")
    except Exception as e:
        logger.error(f"Error in daily retrain job: {e}", exc_info=True)
    finally:
        training_in_progress = False # Release flag

def weekly_full_retrain_job():
    """ Retrain full model từ base_weather.csv mỗi tuần """
    if not TRAINING_ENABLED:
        return
    logger.info("Running weekly full retrain...")
    try:
        from core.training import full_train_from_csv
        full_train_from_csv()
    except Exception as e:
        logger.error(f"Error in weekly retrain: {e}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    if TRAINING_ENABLED:
        database.connect_to_mongo()

    if prediction.model is None:
        logger.warning("Model was not loaded successfully on initial import. Predictions may fail.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
    if TRAINING_ENABLED:
        database.close_mongo_connection()
    logger.info("Shutdown complete.")

if scheduler:
    atexit.register(scheduler.shutdown)
if TRAINING_ENABLED:
    atexit.register(database.close_mongo_connection)

# --- Health Check ---
@app.get("/health", status_code=status.HTTP_200_OK, summary="Check API Health Status")
async def health_check():
    """Checks database connection and model loading."""
    db_ok = None
    if TRAINING_ENABLED:
        db_ok = False
        if database.mongo_client is not None and database.db is not None:
            try:
                database.mongo_client.admin.command('ping')
                db_ok = True
            except errors.ConnectionFailure:
                logger.warning("Health Check: DB ping failed.")
            except Exception as e:
                logger.warning(f"Health Check: DB check error - {e}")
    else:
        db_ok = "N/A (Training Disabled)"

    model_ok = (
        prediction.model is not None and
        prediction.label_encoder is not None and
        prediction.scaler is not None
    )

    status_detail = {
        "database_connection": "ok" if db_ok is True else ("error" if db_ok is False else db_ok),
        "prediction_model_loaded": "ok" if model_ok else "error"
    }

    if model_ok and (db_ok is True or db_ok == "N/A (Training Disabled)"):
        return {"status": "ok", "details": status_detail}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy. Details: {status_detail}"
        )

# --- Background retrain and fetch
if TRAINING_ENABLED:
    def check_if_training_running() -> bool:
        """Checks if any task is currently in 'running' state."""
        with tasks_lock:
            return any(task.status == "running" for task in retrain_tasks.values())

    def background_retrain_runner(task_id: str, params: Optional[Dict[str, Any]] = None):
        """Worker function that runs the training and updates status with metrics."""
        global retrain_tasks
        start_time = datetime.now()
        logger.info(f"Starting background retraining task ID: {task_id}")

        # Update status to running
        with tasks_lock:
             if task_id in retrain_tasks:
                  retrain_tasks[task_id].status = "running"
                  retrain_tasks[task_id].start_time = start_time
             else:
                 logger.error(f"Task ID {task_id} not found in tracking dict at start.")
                 return

        task_result = {}
        try:
            # Call the modified train_model function
            task_result = training.train_model(task_id=task_id, hyperparams=params)
            logger.info(f"Training task {task_id} finished with status: {task_result.get('status')}")

        except Exception as e:
            logger.exception(f"Unhandled exception in background_retrain_runner for task {task_id}: {e}")
            task_result = {
                "status": "failed",
                "message": f"Unexpected worker error: {e}",
                "accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_score_macro": None,
                "classification_report_dict": None,
                "params_used": params or {},
                "start_time": start_time,
                "end_time": datetime.now()
            }
        finally:
            # Update final status and results (Modified)
            with tasks_lock:
                if task_id in retrain_tasks:
                    current_task = retrain_tasks[task_id] # Get the RetrainStatus object
                    current_task.status = task_result.get("status", "failed")
                    current_task.message = task_result.get("message", "Worker finished unexpectedly.")
                    # --- Update with new metrics ---
                    current_task.accuracy = task_result.get("accuracy")
                    current_task.precision_macro = task_result.get("precision_macro")
                    current_task.recall_macro = task_result.get("recall_macro")
                    current_task.f1_score_macro = task_result.get("f1_score_macro")
                    current_task.classification_report_dict = task_result.get("classification_report_dict")
                    # --- Keep metadata ---
                    current_task.params_used = task_result.get("params_used")
                    current_task.end_time = task_result.get("end_time", datetime.now())
                    # --- Removed CM URL ---
                    # cm_filename = task_result.get("confusion_matrix_filename")
                    # current_task.confusion_matrix_image_url = f"/static/cm_plots/{cm_filename}" if cm_filename else None
                else:
                     logger.error(f"Task ID {task_id} disappeared from tracking dict before final update.")
    # --- Retrain Endpoints ---

    @app.post("/retrain",
              status_code=status.HTTP_202_ACCEPTED,
              response_model=RetrainTriggerResponse, # Use new model
              summary="Trigger Manual Retraining (Default Params)")
    async def trigger_retrain_default(request: Request, background_tasks: BackgroundTasks):
        """
        Triggers model retraining in the background using default hyperparameters.
        Returns a task ID to monitor the progress.
        """
        if check_if_training_running():
             raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                 detail="Another retraining task is already in progress.")

        task_id = str(uuid.uuid4())
        status_endpoint = str(request.url_for('get_retrain_status', task_id=task_id))

        # Initialize task status
        with tasks_lock:
            retrain_tasks[task_id] = RetrainStatus(
                task_id=task_id,
                status="pending",
                message="Retraining task accepted and queued.",
                status_endpoint=status_endpoint # Store for convenience, though not in model
            )

        # Add the runner to background tasks
        background_tasks.add_task(background_retrain_runner, task_id=task_id, params=None)

        logger.info(f"Accepted default retraining task with ID: {task_id}")
        return RetrainTriggerResponse(
            message="Manual retraining (default params) started in background.",
            task_id=task_id,
            status_endpoint=status_endpoint
        )


    @app.post("/retrain-with-params",
              status_code=status.HTTP_202_ACCEPTED,
              response_model=RetrainTriggerResponse, # Use new model
              summary="Trigger Retraining with Custom Hyperparameters")
    async def trigger_retrain_params(request: Request, params: XGBoostTrainParams, background_tasks: BackgroundTasks):
        """
        Triggers model retraining in the background using specified hyperparameters.
        Returns a task ID to monitor the progress.
        """
        if check_if_training_running():
             raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                 detail="Another retraining task is already in progress.")

        task_id = str(uuid.uuid4())
        status_endpoint = str(request.url_for('get_retrain_status', task_id=task_id))
        params_dict = params.model_dump(exclude_unset=True, by_alias=True) # Use validated params

        # Initialize task status
        with tasks_lock:
             retrain_tasks[task_id] = RetrainStatus(
                 task_id=task_id,
                 status="pending",
                 message="Custom parameter retraining task accepted and queued.",
                 params_used=params_dict # Store requested params initially
             )

        # Add the runner to background tasks
        background_tasks.add_task(background_retrain_runner, task_id=task_id, params=params_dict)

        logger.info(f"Accepted custom param retraining task with ID: {task_id}")
        return RetrainTriggerResponse(
            message="Manual retraining with custom parameters started in the background.",
            task_id=task_id,
            status_endpoint=status_endpoint
        )

    # --- New Status Endpoint ---
    @app.get("/retrain/status/{task_id}",
             response_model=RetrainStatus,
             summary="Get Retraining Task Status and Results")
    async def get_retrain_status(task_id: str):
        """
        Poll this endpoint with the task_id received from a /retrain request
        to check the status and get results (accuracy, CM plot URL) upon completion.
        """
        with tasks_lock:
            task = retrain_tasks.get(task_id)

        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Retraining task with ID '{task_id}' not found.")

        logger.debug(f"Polling status for task {task_id}: Current status is {task.status}")
        return task

    @app.post("/fetch-data", status_code=status.HTTP_202_ACCEPTED, response_model=BackgroundTaskResponse, summary="Trigger manual data fetch")
    async def fetch_data_endpoint(background_tasks: BackgroundTasks, start_date_str: str, end_date_str: str):
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            if start_date > end_date:
                raise HTTPException(status_code=400, detail="Start date must be before or equal to end date.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        background_tasks.add_task(data_loader.fetch_and_store_data_for_provinces, start_date, end_date)
        return BackgroundTaskResponse(message=f"Fetching data from {start_date_str} to {end_date_str} started.")

# Get Current Weather API
@app.get("/weather/current", response_model=CurrentWeatherResponse, summary="Get Current Weather by Coordinates")
async def get_current_weather_endpoint(
    latitude: float = Query(..., description="Latitude of the location", example=10.7626),
    longitude: float = Query(..., description="Longitude of the location", example=106.6601)
):
    """
    Fetches the current weather conditions for the given latitude and longitude
    from WorldWeatherOnline. If the API doesn't provide a clear weather description,
    it falls back to using the trained prediction model.
    """
    logger.info(f"Received request for current weather at lat={latitude}, lon={longitude}")

    # Ensure model is loaded if fallback might be needed
    if prediction.model is None or prediction.label_encoder is None or prediction.scaler is None:
         logger.warning("Prediction model/preprocessors not loaded, fallback prediction will not be available.")
         # Continue fetching from API, but prediction fallback won't work

    weather_info = current_weather.get_current_weather(latitude, longitude)

    if weather_info is None:
        # Error logged within get_current_weather
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Failed to retrieve or parse current weather data from the external API.")

    # The get_current_weather function now includes the fallback logic
    # and adds 'condition_source'
    return CurrentWeatherResponse(**weather_info)

# --- Predict
@app.post("/predict", response_model=PredictResponse, summary="Get Weather Predictions")
async def predict_endpoint(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No instances provided.")

    try:
        input_df = pd.DataFrame([item.model_dump() for item in request.instances])
        if 'datetime' in input_df.columns and input_df['datetime'].notna().any():
            input_df['datetime'] = pd.to_datetime(input_df['datetime'])
    except Exception as e:
        logger.error(f"Invalid input format: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if prediction.model is None or prediction.label_encoder is None or prediction.scaler is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                             detail="Prediction service unavailable: Model/Preprocessors not loaded.")

    results = prediction.predict_weather(input_df)

    if not isinstance(results, list) or not results or any("Error:" in str(res) for res in results):
        logger.error(f"Prediction failed: {results}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {results}")

    return PredictResponse(predictions=results)

# --- Root info
@app.get("/", summary="API Root / Info", include_in_schema=False)
async def read_root():
    return {"message": "Vietnam Weather Forecast API. Use /docs for details."}
