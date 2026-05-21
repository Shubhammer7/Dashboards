import pandas as pd


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for c in ["Sales", "Profit", "Quantity"]:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "Order Date" in df.columns:
        df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df
