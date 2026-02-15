def engineer_features(df):
    df["Return"] = df["Close"].pct_change()
    df["Rolling_Mean_20"] = df["Close"].rolling(20).mean()
    df["Rolling_STD_20"] = df["Close"].rolling(20).std()
    df = df.dropna()
    return df