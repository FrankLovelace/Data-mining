# %%
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ruta_limpia = '../data/MX_videos_limpio.csv'
df = pd.read_csv(ruta_limpia)

print("Dataset listo para análisis de texto.")

# %%
text = ' '.join(df['title'].dropna())

print("Se han combinado todos los títulos en un solo bloque de texto.")

# %%
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)

print("Nube de palabras generada.")


# %%
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.title('Nube de Palabras de Títulos de Videos en Tendencia en México', fontsize=20)
plt.show()

# %%
stopwords_es = set([
    'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'las', 'un', 'por', 'con', 'no', 'una',
    'su', 'para', 'es', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'he', 'mi', 'sin',
    'qué', 'me', 'este', 'ya', 'o', 'se', 'ha', 'que'
])

wordcloud_clean = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stopwords_es,
    collocations=False
).generate(text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud_clean, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras (sin Stopwords comunes)', fontsize=20)
plt.show()


