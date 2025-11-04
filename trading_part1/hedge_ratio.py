#find hedge ratio via linear regression
import pandas as pd
import statsmodels.api as sm

def calculate_spread(df, ticker1, ticker2, hedge_ratio):
    
    spread = df[ticker1] - (hedge_ratio * df[ticker2])
    return spread

def rolling_mean_calc(spread, windows=(30, 60, 252)):

    table = pd.DataFrame({"spread": spread})

    for w in windows:
        table[f"{w}daymean"] = spread.rolling(w, min_periods=w).mean()
    return table

def rolling_std_calc(spread, windows=(30, 60, 252)):

    table = pd.DataFrame({"spread": spread})

    for w in windows:
        table[f"{w}daystd"]  = spread.rolling(w, min_periods=w).std()
    return table

# load the file and read date column
df = pd.read_csv("data/KO_PEP.csv", parse_dates=["Date"])
# sort by date
df = df.set_index("Date").sort_index()
# drop missing values
df = df[["KO", "PEP"]].dropna()

# dependent variable 
y = df["KO"]
# independent variable, need the sm.add_constant or else it would go through the origin and have no intercepts
X = sm.add_constant(df["PEP"])

# Build and fit the OLS regression
model = sm.OLS(y, X)
results = model.fit()

# Pull out the coefficients
alpha = results.params["const"]
# stores hedge ratio
beta  = results.params["PEP"]
print(f'Hedge Ratio: {beta}')

# print regression analysis
print(results.summary())     

# spread
spread = calculate_spread(df, "KO", "PEP", beta)
# rolling data
rolling_mean = rolling_mean_calc(spread, windows=(30, 60, 252))
rolling_std = rolling_std_calc(spread, windows=(30, 60, 252))