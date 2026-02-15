import pandas as pd
from pathlib import Path

def load_data(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df