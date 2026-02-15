import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.features.build_features import engineer_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model

def run_training_pipeline():

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    df = load_data(config["data"]["path"])
    df = clean_data(df)
    df = engineer_features(df)

    target = config["data"]["target"]

    # Remove target and Date
    X = df.drop(columns=[target, "Date"], errors="ignore")
    y = df[target]

    # Remove any non-numeric columns (fix for XGBoost)
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], shuffle=False
    )

    model = train_model(X_train, y_train, config)
    metrics = evaluate_model(model, X_test, y_test)

    output_path = Path(config["output"]["model_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print("Training Complete")
    print(metrics)
