import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

sales = np.array([100,110,140,180,150,520,220,320,960,790])

# Скользящее среднее

def moving_average(series, window=3):
    if len(series) < window:
        return np.mean(series)
    return np.mean(series[-window])
ma_pred = moving_average(sales, window=3)
print(f"Moving average prediction to next week: {ma_pred}")

# Линейная регрессия

x = np.arange(len(sales)).reshape(-1,1)     # Недели (0,1,2,3)
y = sales

model = LinearRegression()
model.fit(x,y)

next_week = np.array([[len(sales)]])
linear_pred = model.predict(next_week)[0]
print(f"Линейная регрессия предикт на следующую неделю: {linear_pred}")
