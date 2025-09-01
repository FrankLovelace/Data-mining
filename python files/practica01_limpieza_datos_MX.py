# %%
import pandas as pd
import json


# %%

ruta_videos = '../data/MXvideos.csv'
ruta_categorias = '../data/MX_category_id.json'
try:
    df_videos = pd.read_csv(ruta_videos)
except UnicodeDecodeError:
    print("Error de codificación UTF-8, intentando con 'latin1'...")
    df_videos = pd.read_csv(ruta_videos, encoding='latin1')

with open(ruta_categorias, 'r') as f:
    data_json = json.load(f)

categorias = {}
for item in data_json['items']:
    categorias[int(item['id'])] = item['snippet']['title']

print("Datos de México cargados exitosamente.")

# %%
print("Primeras filas del dataset de videos de México:")
df_videos.head()

# %%
print("\nInformación general del DataFrame:")
df_videos.info()

# %%
print("\nConteo de valores nulos por columna:")
df_videos.isnull().sum()

# %%

df_videos['trending_date'] = pd.to_datetime(df_videos['trending_date'], format='%y.%d.%m')

df_videos['publish_date'] = pd.to_datetime(df_videos['publish_time']).dt.date
df_videos['publish_date'] = pd.to_datetime(df_videos['publish_date'])

df_videos['category_name'] = df_videos['category_id'].map(categorias)

df_videos['description'] = df_videos['description'].fillna('')

print("¡Limpieza de datos completada!")

# %%
print("Información del DataFrame después de la limpieza:")
df_videos.info()

print("\nNuevos conteos de nulos:")
df_videos.isnull().sum()

# %%
df_videos['category_name'].fillna('Categoría Desconocida', inplace=True)

print("Valores nulos en 'category_name' rellenados.")

# %%
ruta_limpia = '../data/MX_videos_limpio.csv'
df_videos.to_csv(ruta_limpia, index=False)

print(f"Dataset limpio de México guardado exitosamente en: {ruta_limpia}")

# %%
print("Conteo final de valores nulos:")
df_videos.isnull().sum()


