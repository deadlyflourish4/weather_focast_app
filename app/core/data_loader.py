import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import logging
import urllib.parse
import json
import time # Import time for potential delays
from .config import settings
from .preprocessing import handle_description
from .database import save_data_to_mongo # Import the save function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_province_coordinates():
    """Loads Vietnam province names and coordinates from a JSON file."""
    try:
        logger.info(f"Loading province data from JSON: {settings.PROVINCES_DATA_PATH}")
        with open(settings.PROVINCES_DATA_PATH, 'r', encoding='utf-8') as f:
            province_list = json.load(f)
        df = pd.DataFrame(province_list)
        if not all(col in df.columns for col in ['province_name', 'latitude', 'longitude']):
             logger.error(f"Provinces JSON missing required keys.")
             return pd.DataFrame(columns=['province_name', 'latitude', 'longitude'])
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        logger.info(f"Loaded {len(df)} provinces from JSON.")
        return df
    except FileNotFoundError:
        logger.error(f"Province data file not found: {settings.PROVINCES_DATA_PATH}")
        return pd.DataFrame(columns=['province_name', 'latitude', 'longitude'])
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {settings.PROVINCES_DATA_PATH}: {e}")
        return pd.DataFrame(columns=['province_name', 'latitude', 'longitude'])
    except Exception as e:
        logger.error(f"Error loading province data from JSON: {e}")
        return pd.DataFrame(columns=['province_name', 'latitude', 'longitude'])


def call_api_to_get_data(date_str, enddate_str, city_name_raw):
    """Fetches historical data from WorldWeatherOnline using city name."""
    try:
        city_encoded = urllib.parse.quote_plus(city_name_raw)
        api_key = settings.WORLD_WEATHER_API_KEY
        if not api_key or api_key == "YOUR_DEFAULT_KEY":
            logger.error("WorldWeatherOnline API key is not configured.")
            return None

        link = f"http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q={city_encoded}&date={date_str}&enddate={enddate_str}&key={api_key}&format=json&tp=1"
        logger.debug(f"Requesting API: {link.replace(api_key, '***')}")
        response = requests.get(link, timeout=45) # Increased timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out for city '{city_name_raw}' ({date_str}-{enddate_str})")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for city '{city_name_raw}' ({date_str}-{enddate_str}): {e}")
        return None
    except Exception as e:
         logger.error(f"Error processing API response for city '{city_name_raw}': {e}")
         return None


def parse_api_data(json_data, province_name, lat, lon):
    """Parses the JSON response into structured lists."""
    times, humidities, pressures, temps, wind_dirs, wind_speeds, descs = [], [], [], [], [], [], []
    if not json_data or 'data' not in json_data or 'weather' not in json_data['data']:
        logger.warning(f"No valid data found in API response for {province_name}. Response: {json_data}")
        return pd.DataFrame({ 'datetime': [], 'humidity': [], 'temperature': [], 'pressure': [], 'wind_direction': [], 'wind_speed': [], settings.LABEL_COL: [], 'province': [], 'latitude': [], 'longitude': []})

    try:
        for day_data in json_data["data"]["weather"]:
            date = day_data["date"]
            for hour_data in day_data.get('hourly', []):
                time_val = hour_data.get("time")
                if time_val is None: continue
                hour_str = f"{int(time_val) // 100:02d}"
                try:
                    time_dt = datetime.strptime(f"{date} {hour_str}:00:00", "%Y-%m-%d %H:%M:%S")
                    times.append(time_dt)
                except ValueError:
                    logger.warning(f"Could not parse time: {date} {hour_str} for {province_name}")
                    continue

                humidities.append(float(hour_data.get("humidity", np.nan)))
                pressures.append(float(hour_data.get("pressure", np.nan)))
                temps.append(float(hour_data.get("tempC", np.nan)) + 273.15)
                wind_dirs.append(float(hour_data.get("winddirDegree", np.nan)))
                ws_kmph_str = hour_data.get("windspeedKmph", np.nan)
                try: wind_speeds.append(round(float(ws_kmph_str) * (10/36), 2))
                except (ValueError, TypeError): wind_speeds.append(np.nan)
                weather_desc = hour_data.get("weatherDesc", [{}])[0].get("value", "Unknown")
                descs.append(handle_description(weather_desc))

        df = pd.DataFrame({'datetime': times, 'humidity': humidities, 'temperature': temps, 'pressure': pressures, 'wind_direction': wind_dirs, 'wind_speed': wind_speeds, settings.LABEL_COL: descs, 'province': [province_name] * len(times), 'latitude': [lat] * len(times), 'longitude': [lon] * len(times)})
        return df
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        logger.error(f"Error parsing API data structure for {province_name}: {e}")
        logger.debug(f"Problematic JSON snippet (day data): {day_data}")
        return pd.DataFrame({'datetime': [], 'humidity': [], 'temperature': [], 'pressure': [], 'wind_direction': [], 'wind_speed': [], 'weather_condition': [], 'province': [], 'latitude': [], 'longitude': []})
    except Exception as e:
        logger.error(f"Unexpected error during API data parsing for {province_name}: {e}")
        return pd.DataFrame({'datetime': [], 'humidity': [], 'temperature': [], 'pressure': [], 'wind_direction': [], 'wind_speed': [], 'weather_condition': [], 'province': [], 'latitude': [], 'longitude': []})


def get_lst_first_day_last_day(start_date, end_date):
    """Generates tuples of (first_day_of_month, last_day_of_month) strings."""
    date_format = "%Y-%m-%d"
    dates_list = []
    current_date = start_date.replace(day=1) # Start from beginning of month
    while current_date <= end_date:
        last_day_num = calendar.monthrange(current_date.year, current_date.month)[1]
        first_day_of_month = current_date.replace(day=1)
        last_day_of_month = current_date.replace(day=last_day_num)

        actual_start_day = max(first_day_of_month, start_date) # Don't start before requested start
        actual_last_day = min(last_day_of_month, end_date)     # Don't go past requested end

        if actual_start_day <= actual_last_day: # Ensure range is valid
            dates_list.append((actual_start_day.strftime(date_format), actual_last_day.strftime(date_format)))

        # Move to the next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1, day=1)

    return dates_list


def fetch_and_store_data_for_provinces(start_date: datetime, end_date: datetime) -> int:
    """
    Fetches new data for all VN provinces within the date range
    and saves it directly to MongoDB. Adds delays to respect API limits.
    Returns the number of new/updated records saved.
    """
    provinces_df = get_province_coordinates()
    date_ranges = get_lst_first_day_last_day(start_date, end_date)
    total_saved_count = 0
    api_call_delay = 1.1 # Seconds between API calls (adjust based on WWO limits/plan)

    if provinces_df.empty:
        logger.error("No province data available to fetch.")
        return 0

    total_provinces = len(provinces_df)
    for i, (index, row) in enumerate(provinces_df.iterrows()):
        province_name_raw = row['province_name']
        lat = row['latitude']
        lon = row['longitude']
        logger.info(f"Fetching & storing data for province {i+1}/{total_provinces}: {province_name_raw}...")

        province_data_frames = []
        for date_tuple in date_ranges:
            start_str, end_str = date_tuple
            logger.debug(f"  Fetching range: {start_str} to {end_str} for {province_name_raw}")
            json_data = call_api_to_get_data(start_str, end_str, province_name_raw)
            time.sleep(api_call_delay) # Add delay between API calls

            if json_data:
                df_parsed = parse_api_data(json_data, province_name_raw, lat, lon)
                if not df_parsed.empty:
                    province_data_frames.append(df_parsed)
            else:
                 logger.warning(f"  Skipping range due to API error: {start_str} to {end_str} for {province_name_raw}")

        if province_data_frames:
            province_df_combined = pd.concat(province_data_frames, ignore_index=True)
            logger.info(f"  Attempting to save {len(province_df_combined)} records for {province_name_raw} to MongoDB...")
            saved_count = save_data_to_mongo(province_df_combined)
            total_saved_count += saved_count
            logger.info(f"  Saved {saved_count} new/updated records for {province_name_raw}.")
        else:
            logger.warning(f"No data fetched for province: {province_name_raw}, nothing to save.")

    logger.info(f"Finished fetching and storing data. Total new/updated records saved: {total_saved_count}")
    return total_saved_count

