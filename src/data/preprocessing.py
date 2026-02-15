def clean_data(df):
    df = df.ffill()
    df = df.dropna()
    return df
