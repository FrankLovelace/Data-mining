# %%
import pandas as pd

# ola
pd.set_option('display.float_format', lambda x: '%.2f' % x)

ruta_limpia = '../data/MX_videos_limpio.csv'
df = pd.read_csv(ruta_limpia)

print("Dataset limpio cargado. Aquí están las primeras 5 filas:")
df.head()

# %%
print("Estadísticas descriptivas de las columnas numéricas:")
df.describe()

# %%
print("Estadísticas descriptivas de las columnas categóricas:")
df.describe(include=['object'])

# %%
stats_by_category = df.groupby('category_name')[['views', 'likes', 'dislikes', 'comment_count']].agg(['mean', 'sum', 'max'])

print("Estadísticas agrupadas por categoría de video:")
stats_by_category

# %%
stats_ordenadas_por_vistas = stats_by_category.sort_values(by=('views', 'mean'), ascending=False)

print("Estadísticas por categoría, ordenadas por el PROMEDIO de vistas:")
stats_ordenadas_por_vistas


