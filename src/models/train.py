from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_model(X_train, y_train, config):
    model_type = config["model"]["type"]

    if model_type == "xgboost":
        model = XGBRegressor(random_state=config["model"]["random_state"])
    else:
        model = RandomForestRegressor(random_state=config["model"]["random_state"])

    model.fit(X_train, y_train)
    return model