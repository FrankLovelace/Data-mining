# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ruta_limpia = '../data/MX_videos_limpio.csv'
df = pd.read_csv(ruta_limpia)

features = ['views', 'likes', 'dislikes', 'comment_count']
X = df[features]

print("Dataset listo para clustering.")

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Datos escalados y listos para el modelo K-Means.")

# %%
inertia = []
k_range = range(1, 11) 

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Encontrar k Óptimo')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# %%
k_optimo = 4

kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['cluster'] = kmeans.labels_

print(f"Modelo K-Means entrenado con {k_optimo} clusters.")
print("Se ha añadido una columna 'cluster' al DataFrame.")

# %%
cluster_analysis = df.groupby('cluster')[features].mean()
print("Análisis de las características promedio de cada cluster:")
print(cluster_analysis)

category_distribution = df.groupby('cluster')['category_name'].value_counts(normalize=True).unstack().fillna(0)
print("\nDistribución de categorías reales dentro de cada cluster (en %):")
print(category_distribution.head())

# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='views', y='likes', hue='cluster', palette='viridis', alpha=0.6)
plt.title('Visualización de Clusters: Vistas vs. Likes')
plt.xlabel('Vistas (escala log)')
plt.ylabel('Likes (escala log)')
plt.xscale('log')
plt.yscale('log')
plt.legend(title='Cluster')
plt.show()


