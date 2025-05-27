# Vietnam Weather Forecast API

## Overview

This project provides a FastAPI application for predicting weather conditions across various provinces in Vietnam. It utilizes an XGBoost machine learning model trained on historical weather data. The application includes features for:

*   **Prediction:** Serving weather condition predictions based on input features (humidity, temperature, pressure, wind, location).
*   **Data Persistence:** Storing fetched weather data in a MongoDB database.
*   **Automated Data Fetching:** A scheduled daily job to fetch the latest historical data for Vietnamese provinces from the WorldWeatherOnline API and store it in MongoDB.
*   **Automated Retraining:** A scheduled daily job to retrain the XGBoost model using all the data accumulated in the MongoDB database.
*   **Manual Retraining:** API endpoints to trigger retraining manually, either with default hyperparameters or with custom ones provided in the request.
*   **Manual Data Fetching:** An API endpoint to manually fetch historical data for a specified date range and store it in MongoDB.

## Features

*   FastAPI backend providing RESTful endpoints.
*   XGBoost model for multi-class weather condition classification.
*   Integration with WorldWeatherOnline API for historical data fetching.
*   MongoDB integration for storing historical and fetched weather data.
*   APScheduler for automating daily data fetching and model retraining.
*   Configurable hyperparameters for retraining via API.
*   Preprocessing pipeline using Scikit-learn (Label Encoding, Scaling).
*   Undersampling technique applied during training to handle imbalanced data (using the logic from `train_pipeline.py`).
*   Model, Scaler, and Label Encoder persistence using `joblib`.
*   Configuration management via `.env` file.
*   Basic health check endpoint.
*   Interactive API documentation via Swagger UI (`/docs`).

## Tech Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **ML Model:** XGBoost (via `xgboost` library)
*   **Data Processing/Handling:** Pandas, Scikit-learn, Numpy
*   **Database:** MongoDB (via `pymongo`)
*   **Scheduling:** APScheduler
*   **Configuration:** python-dotenv
*   **Serialization:** joblib
*   **Data Source (External):** WorldWeatherOnline API
*   **Containerization (DB):** Docker, Docker Compose

## Prerequisites

*   Python 3.8+ and Pip
*   Docker and Docker Compose installed ([Docker Installation Guide](https://docs.docker.com/get-docker/))
*   Git (for cloning the repository)
*   A valid API Key from [WorldWeatherOnline](https://www.worldweatheronline.com/)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd weather_forecast_api
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your actual details:
        *   `WORLD_WEATHER_API_KEY`: **Required**. Your key from WorldWeatherOnline.
        *   `MONGO_URI`: **Required**. Your MongoDB connection string (e.g., `mongodb://localhost:27017/` for default local Docker setup).
        *   Adjust `MONGO_DB_NAME` and `MONGO_COLLECTION_NAME` if needed.
        *   Adjust scheduler hours (`DAILY_FETCH_HOUR`, `DAILY_RETRAIN_HOUR`) if desired.

5.  **Prepare Province Data:**
    *   Ensure the `data/` directory exists.
    *   Create or ensure the `data/provinces_vn.json` file exists and contains the list of Vietnamese provinces with their `province_name`, `latitude`, and `longitude`. Make sure the `province_name` values are recognizable by the WorldWeatherOnline API.

6.  **Prepare Historical Data CSVs:**
    *   Ensure the `data/` directory contains the following CSV files from the `historical-hourly-weather-dataset`:
        *   `humidity.csv`
        *   `pressure.csv`
        *   `temperature.csv`
        *   `weather_description.csv`
        *   `wind_direction.csv`
        *   `wind_speed.csv`
        *   `city_attributes.csv`

## Running the Application Step-by-Step

1.  **Start MongoDB using Docker Compose:**
    *   Make sure Docker Desktop (or Docker Engine + Compose) is running.
    *   (If you don't have one, create a `docker-compose.yml` file in the project root):
      ```yaml
      version: '3.7'
        services:
          mongodb_container:
            image: mongo:latest
            ports:
              - 27017:27017
            volumes:
              - mongodb_data_container:/data/db
        
        volumes:
          mongodb_data_container:
      ```
    *   From the project root directory (`weather_forecast_api/`), run:
        ```bash
        docker-compose up -d
        ```
        This command starts a MongoDB container in the background. The `-d` flag means detached mode.

2.  **Run Initial Model Training:**
    *   The API needs pre-trained model files (`.pkl`) to start serving predictions. Run the standalone training script **once** to create these from your historical CSV data.
    *   Make sure your virtual environment is activated.
    *   From the project root directory, run:
        ```bash
        python train_pipeline.py
        ```
    *   This will process the CSVs, train the model with undersampling, perform an evaluation, and save `xgboost_weather.pkl`, `label_encoder.pkl`, and `scaler.pkl` into the `models/` directory. Check the output for success messages and evaluation metrics.

3.  **Run the FastAPI Application:**
    *   Make sure your virtual environment is activated.
    *   From the project root directory, run:
        ```bash
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
        ```
    *   The server will start. Check the logs to ensure it connects to MongoDB and successfully loads the model/preprocessors saved in the previous step.
    *   It will also attempt to fetch recent data if the database is empty (controlled by `.env`) and start the daily background jobs.

4.  **Access the API:**
    *   Open your browser to **`http://localhost:8000/docs`** (or `http://<your-ip>:8000/docs`) to see the interactive Swagger UI documentation.
    *   Test the `/health` endpoint.
    *   Use the `/predict` endpoint to get weather predictions.
    *   Use `/fetch-data` and `/retrain` or `/retrain-with-params` to manually manage data and models.

## API Endpoints

*   **`GET /health`**: Checks the status of the API, database connection, and model loading.
*   **`POST /predict`**: Takes a list of feature instances and returns predicted weather conditions. See `/docs` for the required request body schema (`PredictRequest`).
*   **`POST /fetch-data`**: (Requires `TRAINING_ENABLED`) Triggers a background task to fetch historical weather data for a given date range (`start_date_str`, `end_date_str` in YYYY-MM-DD format) and stores it in MongoDB.
*   **`POST /retrain`**: (Requires `TRAINING_ENABLED`) Triggers a background task to retrain the model using **all** data currently in MongoDB with **default** hyperparameters.
*   **`POST /retrain-with-params`**: (Requires `TRAINING_ENABLED`) Triggers a background task to retrain the model using **all** data currently in MongoDB with **custom** XGBoost hyperparameters provided in the request body. See `/docs` for the `XGBoostTrainParams` schema.

## Configuration (`.env`)

Modify the `.env` file to configure:

*   `WORLD_WEATHER_API_KEY`: Your API key.
*   `MONGO_URI`: Connection string for your MongoDB instance.
*   `MONGO_DB_NAME`, `MONGO_COLLECTION_NAME`: Database and collection names.
*   `PROVINCES_DATA_PATH`: Path to your province JSON file.
*   `MODEL_PATH`, `LABEL_ENCODER_PATH`, `SCALER_PATH`: Paths where model artifacts are saved/loaded.
*   `DAILY_FETCH_HOUR`, `DAILY_RETRAIN_HOUR`: Times (0-23) for scheduled jobs.
*   `FETCH_RECENT_ON_EMPTY_DB`, `FETCH_RECENT_DAYS`: Control initial data fetching on startup if DB is empty.

## Training Details

*   **Initial Training:** Run `python train_pipeline.py` manually. This uses the historical CSV data, applies undersampling to the training portion, trains the model, evaluates it on the original test split, and saves the model/scaler/encoder.
*   **Retraining (API/Scheduled):** The `/retrain` endpoint, `/retrain-with-params` endpoint, and the scheduled `daily_retrain_job` load *all* data from the MongoDB collection, preprocess it (fitting *new* scalers/encoders), split it, train a *new* model, save the artifacts, and then hot-reload the model for the `/predict` endpoint.
