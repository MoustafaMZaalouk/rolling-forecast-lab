"""Small, reproducible educational experiment. See README for limitations."""
import json
from pathlib import Path
import numpy as np

def save(result, output):
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def series(n=420,seed=42):
    rng=np.random.default_rng(seed); t=np.arange(n)
    return 100+.045*t+12*np.sin(2*np.pi*t/7)+5*np.sin(2*np.pi*t/60)+rng.normal(0,2,n)
def features(y,t):
    return [y[t-1],y[t-2],y[t-7],y[t-14],float(np.mean(y[t-7:t])),np.sin(2*np.pi*t/7),np.cos(2*np.pi*t/7),t]
def run(output="results.json"):
    y=series(); folds=[]
    for boundary,end in [(210,280),(280,350),(350,420)]:
        X=np.array([features(y,t) for t in range(14,boundary)]); target=y[14:boundary]
        model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(X,target)
        # One-step-ahead protocol: prior observed test values are available, never future values.
        preds=model.predict(np.array([features(y,t) for t in range(boundary,end)]))
        actual=y[boundary:end]
        folds.append({"train_end_exclusive":boundary,"test_end_exclusive":end,"test_rows":end-boundary,"last_value_mae":float(mean_absolute_error(actual,y[boundary-1:end-1])),"weekly_mae":float(mean_absolute_error(actual,y[boundary-7:end-7])),"ridge_mae":float(mean_absolute_error(actual,preds))})
    return save({"seed":42,"rows":len(y),"protocol":"expanding training windows; frozen model per fold; one-step-ahead observed lags","folds":folds,"mean_ridge_mae":float(np.mean([r['ridge_mae'] for r in folds])),"mean_weekly_mae":float(np.mean([r['weekly_mae'] for r in folds]))},output)

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
