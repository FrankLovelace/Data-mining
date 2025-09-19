# %%
import pandas as pd
from scipy import stats 

ruta_limpia = '../data/MX_videos_limpio.csv'
df = pd.read_csv(ruta_limpia)

print("Dataset listo para las pruebas estadísticas.")
df.head()

# %%


# %%
# Asegura que 'df' y 'stats' estén disponibles aunque esta celda se ejecute antes de la celda 0
if 'df' not in globals():
    if 'pd' not in globals():
        import pandas as pd
    if 'ruta_limpia' in globals():
        df = pd.read_csv(ruta_limpia)
    else:
        ruta_limpia = '../data/MX_videos_limpio.csv'
        df = pd.read_csv(ruta_limpia)

if 'stats' not in globals():
    from scipy import stats

sample_views = df['views'].sample(n=5000, random_state=42)

# Realizamos el Test de Shapiro-Wilk
shapiro_stat, shapiro_p_value = stats.shapiro(sample_views)

print(f"Resultados del Test de Shapiro-Wilk para la distribución de 'views':")
print(f"Estadístico W: {shapiro_stat:.4f}")
print(f"P-valor: {shapiro_p_value}")

# Interpretamos el resultado
alpha = 0.05
if shapiro_p_value < alpha:
    print(f"\nConclusión: El P-valor ({shapiro_p_value:.4g}) es menor que {alpha}.")
    print("Rechazamos la Hipótesis Nula (H0).")
    print(">> La distribución de los datos NO es normal.")
else:
    print(f"\nConclusión: El P-valor ({shapiro_p_value:.4g}) es mayor que {alpha}.")
    print("No podemos rechazar la Hipótesis Nula (H0).")
    print(">> La distribución de los datos podría ser normal.")

# %%
top_5_categories = df['category_name'].value_counts().nlargest(5).index
print(f"Las 5 categorías más populares son: {list(top_5_categories)}\n")

df_top5 = df[df['category_name'].isin(top_5_categories)]

samples_by_category = [df_top5['views'][df_top5['category_name'] == category] for category in top_5_categories]

# %%
statistic, p_value = stats.kruskal(*samples_by_category) 

print(f"Resultados de la prueba de Kruskal-Wallis para 'views' entre las 5 categorías principales:")
print(f"Estadístico H: {statistic:.4f}")
print(f"P-valor: {p_value}")

# Interpretamos el resultado
alpha = 0.05  # Nivel de significancia comúnmente usado
if p_value < alpha:
    print(f"\nConclusión: El P-valor ({p_value:.4g}) es menor que {alpha}.")
    print("Rechazamos la Hipótesis Nula (H0).")
    print(">> Existe una diferencia estadísticamente significativa en el número de vistas entre al menos dos de las categorías.")
else:
    print(f"\nConclusión: El P-valor ({p_value:.4g}) es mayor que {alpha}.")
    print("No podemos rechazar la Hipótesis Nula (H0).")
    print(">> No hay evidencia suficiente para afirmar que existe una diferencia en el número de vistas entre las categorías.")


