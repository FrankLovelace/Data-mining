# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ruta_limpia = '../data/MX_videos_limpio.csv'
df = pd.read_csv(ruta_limpia, parse_dates=['trending_date']) # parse_dates para leer la fecha correctamente

time_series_df = df.groupby('trending_date')['views'].sum().reset_index()
time_series_df = time_series_df.rename(columns={'trending_date': 'date', 'views': 'total_views'})

time_series_df = time_series_df.sort_values(by='date')

print("Serie de tiempo creada exitosamente:")
time_series_df.head()

# %%
plt.figure(figsize=(15, 7))
plt.plot(time_series_df['date'], time_series_df['total_views'])
plt.title('Total de Vistas Diarias de Videos en Tendencia en México')
plt.xlabel('Fecha')
plt.ylabel('Total de Vistas')
plt.grid(True)
plt.show()

# %%
time_series_df['time_index'] = (time_series_df['date'] - time_series_df['date'].min()).dt.days

X = time_series_df[['time_index']] 
y = time_series_df['total_views']   

print("Feature 'time_index' creada:")
time_series_df.head()

# %%
model = LinearRegression()
model.fit(X, y)

y_pred_trend = model.predict(X)

print("Modelo de regresión lineal entrenado sobre la serie de tiempo.")

# %%
last_time_index = X.iloc[-1]['time_index']
future_time_index = np.array(range(last_time_index + 1, last_time_index + 31)).reshape(-1, 1)

future_predictions = model.predict(future_time_index)

last_date = time_series_df['date'].max()
future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 31)])
future_df = pd.DataFrame({'date': future_dates, 'predicted_views': future_predictions})

print("Predicciones para los próximos 30 días:")
print(future_df)

# %%
plt.figure(figsize=(15, 7))

plt.plot(time_series_df['date'], y, label='Datos Reales')
plt.plot(time_series_df['date'], y_pred_trend, color='orange', linestyle='--', label='Línea de Tendencia del Modelo')
plt.plot(future_df['date'], future_df['predicted_views'], color='red', linestyle='--', marker='o', label='Pronóstico a 30 días')

plt.title('Pronóstico de Vistas Totales Diarias')
plt.xlabel('Fecha')
plt.ylabel('Total de Vistas')
plt.legend()
plt.grid(True)
plt.show()


