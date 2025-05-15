import os
from pathlib import Path
import datetime
import time

from .constants import DB_SAVE_INTERVAL, DEBUG

def find_latest_csv(base_dir):
    """
    Finds the latest CSV file in the given directory and subdirectories.

    Parameters
    ----------
    base_dir : str
        The base directory to search for CSV files.

    Returns
    -------
    str
        The path to the latest CSV file, or None if no files are found.
    """
    base_path = Path(base_dir)

    # Recursively glob all CSV files in dir + subdirs
    csv_files = base_path.rglob('*.csv')

    try:
        # Get the most recent file based on modification time
        latest_file = max(csv_files, key=os.path.getmtime)
        return str(latest_file)

    except ValueError:
        # Raised by max() if no files are found
        return None


def save_data_to_database(dataframe, folder_path, last_save_time):
    """
    Saves the DataFrame to a timestamped CSV in 'folder_path/data_archive'
    if at least DB_SAVE_INTERVAL seconds have passed since the last save.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to save.
    folder_path : str
        The folder path where the CSV will be saved.
    last_save_time : float
        The last time the database was saved (in seconds since epoch).
    
    Returns
    -------
    float
        The updated last save time.
    """
    current_time = time.time()

    # Only save every DB_SAVE_INTERVAL seconds
    if current_time - last_save_time < DB_SAVE_INTERVAL:
        return last_save_time

    # Create a database folder 'data_archive' if it doesn't exist
    archive_folder = Path(folder_path) / "data_archive"
    archive_folder.mkdir(parents=True, exist_ok=True)

    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = archive_folder / f"data_archive_{timestamp}.csv"

    try:
        # Save the DataFrame to a CSV file
        dataframe.to_csv(csv_path, index=False)
        if DEBUG: print(f"[DEBUG] Saved {len(dataframe)} rows to {csv_path}")
        return current_time

    except Exception as e:
        if DEBUG: print(f"[DEBUG] Error saving data: {str(e)}")
        return last_save_time

