import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSmodels_AVAILABLE = True
except Exception:
    STATSmodels_AVAILABLE = False


def forecast_monthly(ts, periods=30):
    ts = ts.asfreq('MS').fillna(0)
    if STATSmodels_AVAILABLE:
        try:
            model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            fit = model.fit(optimized=True)
            pred = fit.forecast(periods)
            res = fit.resid
            sigma = np.nanstd(res)
            index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
            forecast_df = pd.DataFrame({ 'ds': index, 'yhat': pred.values, 'yhat_lower': pred.values - 1.96*sigma, 'yhat_upper': pred.values + 1.96*sigma })
            return forecast_df
        except Exception:
            pass
    # fallback
    roll = ts.rolling(3).mean().dropna()
    level = roll.iloc[-1] if len(roll) else ts.mean()
    index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
    preds = np.repeat(level, periods)
    resid = ts - ts.rolling(3).mean()
    sigma = np.nanstd(resid.dropna()) if len(resid.dropna())>0 else 0
    forecast_df = pd.DataFrame({ 'ds': index, 'yhat': preds, 'yhat_lower': preds - 1.96*sigma, 'yhat_upper': preds + 1.96*sigma })
    return forecast_df
