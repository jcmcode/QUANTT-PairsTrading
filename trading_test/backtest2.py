#using vectorbt library for backtesting 

import vectorbt as vbt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime


#backtesting strategy with vectorbt instead of custom backtest code