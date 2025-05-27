import requests
import pandas as pd
import numpy as np
import logging
from .config import settings
from .preprocessing import handle_description # For consistent aggregation
from .prediction import predict_weather       # Import prediction function for fallback
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# WorldWeatherOnline endpoint for current conditions
CURRENT_WEATHER_ENDPOINT = "http://api.worldweatheronline.com/premium/v1/weather.ashx"

def get_current_weather_from_api(lat: float, lon: float):
    """Fetches current weather data from WorldWeatherOnline using lat/lon."""
    api_key = settings.WORLD_WEATHER_API_KEY
    if not api_key or api_key == "YOUR_DEFAULT_KEY":
        logger.error("WorldWeatherOnline API key is not configured.")
        return None, "API key not configured"

    params = {
        'q': f"{lat},{lon}",
        'key': api_key,
        'format': 'json',
        'num_of_days': 1 # Required, even for current conditions apparently
    }
    link = CURRENT_WEATHER_ENDPOINT

    try:
        logger.info(f"Requesting current weather for lat={lat}, lon={lon}")
        response = requests.get(link, params=params, timeout=15) # Shorter timeout for current weather
        response.raise_for_status()
        return response.json(), None # Return data, no error message
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out for current weather at {lat},{lon}")
        return None, "API request timed out"
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for current weather at {lat},{lon}: {e}")
        return None, f"API request failed: {e}"
    except Exception as e:
         logger.error(f"Error processing API response for current weather at {lat},{lon}: {e}")
         return None, f"Error processing API response: {e}"

def parse_current_weather_response(api_response: dict, lat: float, lon: float) -> Optional[dict]:
    """Parses the current weather JSON response into a structured dictionary."""
    try:
        if not api_response or 'data' not in api_response or 'current_condition' not in api_response['data'] or not api_response['data']['current_condition']:
            logger.warning(f"API response lacks current_condition data for {lat},{lon}. Response: {api_response}")
            return None

        current = api_response['data']['current_condition'][0]

        # Extract data safely using .get
        temp_c = float(current.get('temp_C', np.nan))
        humidity = float(current.get('humidity', np.nan))
        pressure = float(current.get('pressure', np.nan)) # Usually in mb/hPa
        wind_kmph_str = current.get('windspeedKmph', np.nan)
        wind_dir = float(current.get('winddirDegree', np.nan))
        obs_time = current.get('observation_time', None) # Usually UTC
        weather_desc_list = current.get('weatherDesc', [{}])
        weather_api_desc = weather_desc_list[0].get('value', None) if weather_desc_list else None

        # Calculate derived values
        temp_k = temp_c + 273.15 if not np.isnan(temp_c) else np.nan
        try:
            wind_mps = round(float(wind_kmph_str) * (10/36), 2) if not pd.isna(wind_kmph_str) else np.nan
        except (ValueError, TypeError):
            wind_mps = np.nan

        # Aggregate the description from API
        aggregated_condition = handle_description(weather_api_desc) if weather_api_desc else "unknown"

        processed_data = {
            "latitude": lat,
            "longitude": lon,
            "observed_time_utc": obs_time,
            "temperature_celsius": temp_c,
            "temperature_kelvin": temp_k,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed_mps": wind_mps,
            "wind_direction_degree": wind_dir,
            "weather_condition": aggregated_condition, # Use aggregated condition
            "api_condition_raw": weather_api_desc # Keep raw for debugging if needed
        }

        # Check if essential data for prediction is present (needed for fallback)
        required_for_pred = ['humidity', 'pressure', 'temperature_kelvin', 'wind_direction_degree', 'wind_speed_mps']
        if any(pd.isna(processed_data.get(key)) for key in required_for_pred):
             logger.warning(f"Missing essential features from API response for {lat},{lon} for potential prediction.")
             # Decide if you should still return partial data or None
             # return None # Option: Return None if prediction features are missing
             # Option: Return partial data, prediction fallback will likely fail later if needed
             return processed_data

        return processed_data

    except (KeyError, TypeError, ValueError, IndexError) as e:
        logger.error(f"Error parsing current weather API data structure for {lat},{lon}: {e}")
        logger.debug(f"Problematic JSON snippet (current_condition): {api_response.get('data', {}).get('current_condition', 'Missing')}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during current weather parsing for {lat},{lon}: {e}")
        return None


def get_current_weather(lat: float, lon: float) -> Optional[dict]:
    """
    Main function to get current weather. Fetches from API, parses,
    and uses prediction model as fallback for weather condition.
    """
    api_data, error_msg = get_current_weather_from_api(lat, lon)
    if error_msg:
        # Return error or specific structure? For now, return None and rely on endpoint handling
        logger.error(f"Failed to get data from API: {error_msg}")
        return None # Indicate failure

    weather_data = parse_current_weather_response(api_data, lat, lon)
    if weather_data is None:
        logger.error("Failed to parse weather data from API response.")
        return None # Indicate failure

    condition_source = "api" # Default source

    # --- Fallback Prediction Logic ---
    # Check if the API description was valid or if we should predict
    if weather_data.get('weather_condition') == 'unknown':
        logger.warning(f"API weather description unusable ('{weather_data.get('api_condition_raw')}'). Attempting prediction fallback for {lat},{lon}.")

        # Prepare data for the prediction model (needs specific feature names)
        predict_input = {
            'humidity': weather_data.get('humidity'),
            'pressure': weather_data.get('pressure'),
            'temperature': weather_data.get('temperature_kelvin'), # Use Kelvin
            'wind_direction': weather_data.get('wind_direction_degree'),
            'wind_speed': weather_data.get('wind_speed_mps'), # Use m/s
            'latitude': lat,
            'longitude': lon
        }
        # Check if all required features for prediction are present
        if any(pd.isna(predict_input.get(feat)) for feat in settings.FEATURES):
             logger.error("Cannot run prediction fallback: Missing required features after parsing API response.")
             weather_data['weather_condition'] = "unknown" # Keep it unknown
             condition_source = "api (prediction fallback failed - missing features)"
        else:
            try:
                input_df = pd.DataFrame([predict_input])
                predicted_conditions = predict_weather(input_df) # Call prediction core function

                if predicted_conditions and "Error:" not in predicted_conditions[0]:
                    weather_data['weather_condition'] = predicted_conditions[0]
                    condition_source = "predicted (fallback)"
                    logger.info(f"Fallback prediction successful: {predicted_conditions[0]}")
                else:
                    logger.error(f"Prediction fallback failed: {predicted_conditions}")
                    weather_data['weather_condition'] = "unknown" # Keep it unknown
                    condition_source = "api (prediction fallback failed - prediction error)"
            except Exception as pred_e:
                logger.error(f"Exception during prediction fallback: {pred_e}")
                weather_data['weather_condition'] = "unknown"
                condition_source = "api (prediction fallback failed - exception)"

    # Add the source info to the dictionary
    weather_data['condition_source'] = condition_source
    # Remove raw API description if you don't want it in the final response
    if 'api_condition_raw' in weather_data:
        del weather_data['api_condition_raw']

    return weather_data