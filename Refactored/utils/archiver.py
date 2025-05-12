from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import logging

class DataArchiver:
    def __init__(self, folder_path, save_interval_seconds):
        self.db_folder = Path(folder_path) / "data_archive"
        self.save_interval = save_interval_seconds
        self.last_save_time = 0
        self.db_folder.mkdir(parents=True, exist_ok=True)

    def save_data(self, df: pd.DataFrame):
        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.db_folder / f"data_archive_{timestamp}.csv"

        try:
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved {len(df)} rows to {csv_path}")
            self.last_save_time = current_time
        except Exception as e:
            logging.error(f"Failed to save data: {e}")