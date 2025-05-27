import pandas as pd
from .database import load_data_from_mongo
from .config import settings, BASE_DIR
import logging
from pathlib import Path
from datetime import datetime  # <-- nhớ import thêm

logger = logging.getLogger(__name__)

def merge_today_data_into_csv(today: datetime):
    """
    Merge dữ liệu hôm nay từ MongoDB vào file base_weather.csv
    Chỉ giữ lại các cột cần thiết.
    """
    logger.info(f"🔄 Đang merge dữ liệu ngày {today.strftime('%Y-%m-%d')} vào base_weather.csv...")

    df = load_data_from_mongo()
    if df.empty:
        logger.warning("⚠️ MongoDB không có dữ liệu để merge.")
        return

    # Filter chỉ lấy dữ liệu của đúng ngày cần merge
    df['date_only'] = df['datetime'].dt.date
    today_date = today.date()
    df_today = df[df['date_only'] == today_date]

    if df_today.empty:
        logger.warning(f"⚠️ Không có dữ liệu nào cho ngày {today.strftime('%Y-%m-%d')}. Bỏ qua merge.")
        return

    # Giữ đúng các cột yêu cầu
    columns_to_keep = [
        'datetime', 'province', 'humidity', 'latitude', 'longitude',
        'pressure', 'temperature', 'weather_condition', 'wind_direction', 'wind_speed'
    ]

    df_today = df_today[columns_to_keep]

    # Load base CSV nếu có
    base_csv_path = BASE_DIR / "data" / "base_weather.csv"
    if base_csv_path.exists():
        df_base = pd.read_csv(base_csv_path)
        logger.info(f"📄 Đọc {len(df_base)} records từ base_weather.csv.")
    else:
        df_base = pd.DataFrame(columns=columns_to_keep)
        logger.warning("⚠️ Base CSV chưa tồn tại. Sẽ tạo mới.")

    # Merge
    df_merged = pd.concat([df_base, df_today], ignore_index=True)

    # Remove duplicates
    df_merged.drop_duplicates(subset=["datetime", "province"], keep="last", inplace=True)

    # Save lại
    base_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(base_csv_path, index=False)
    logger.info(f"✅ Merge thành công. Đã lưu {len(df_merged)} records vào base_weather.csv.")
