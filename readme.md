# Análisis de Datos de Tendencias de YouTube México
## 📝 Descripción General

Este repositorio contiene un análisis exhaustivo del conjunto de datos **"Trending YouTube Video Statistics"** para la región de México. El proyecto explora los factores que impulsan la viralidad, el engagement y las tendencias culturales en la plataforma. A través de un proceso que abarca desde la limpieza de datos hasta el machine learning y el análisis estadístico profundo, este trabajo busca responder a la pregunta: ¿Qué patrones se esconden detrás de los videos que capturan la atención de México?
Este proyecto fue desarrollado como parte del curso de Minería de Datos, yendo más allá de las prácticas individuales para consolidar los hallazgos en una investigación cohesiva.

## 💡 Descubrimientos Clave

Mi análisis reveló varios patrones significativos y, en algunos casos, inesperados:

1.  **La Dominancia Categórica es Real y Medible:**
    *   Una prueba de Kruskal-Wallis (p < 0.001) confirmó que esta diferencia de rendimiento no es casualidad, sino un patrón estadísticamente significativo.

2.  **Las Vistas son el Motor del Engagement:**
    *   Existe una fuerte correlación positiva entre las vistas y los likes de un video. Un modelo de regresión lineal demostró que el número de vistas puede predecir el 66.15% de la variabilidad en los likes (R² = 0.6615), estableciéndolo como el predictor más robusto del éxito.

3.  **Los Títulos son un Espejo de la Cultura Popular:**
    *   El análisis de texto de los títulos, a través de una nube de palabras, reveló que el contenido viral está fuertemente anclado a la cultura de masas: programas de TV de alto rating (`Exatlón`, `Enamorándonos`), fenómenos musicales (`BTS`, `Bad Bunny`) y eventos políticos (`AMLO`) dominan la conversación.

4.  **Un Hallazgo Inesperado: El "Engaño" de la Estadística:**
    *   Una profunda investigación sobre los resultados de las pruebas de hipótesis reveló una lección crucial: con datasets masivos, la significancia estadística (p=0.0) no siempre implica una significancia práctica. Descubrimos que diferencias minúsculas en las distribuciones, invisibles al ojo humano, son detectadas por las pruebas, lo que subraya la importancia de evaluar la magnitud del efecto, no solo el p-valor.

## 🛠️ Tecnologías Utilizadas

*   **Lenguaje:** Python
*   **Librerías Principales:**
    *   `pandas` para la manipulación y limpieza de datos.
    *   `matplotlib` y `seaborn` para la visualización de datos.
    *   `scikit-learn` para los modelos de Machine Learning (Regresión Lineal, KNN, K-Means).
    *   `wordcloud` para el análisis de texto.
    *   `scipy` para las pruebas estadísticas.
*   **Entorno:** Jupyter Notebooks a través de VS Code.
*   **Control de Versiones:** Git y GitHub.

## 📂 Estructura del Repositorio
```
data-mining/
├── data/
├── notebooks/
├── PIA/
├── scripts/
└── README.md
```
## 🚀 Cómo Replicar el Análisis

Para ejecutar este proyecto en tu máquina local, sigue estos pasos:

1.  **Clona el repositorio:**
```
    git clone https://github.com/FrankLovelace/Data-mining.git
    cd data-mining
```
2.  **Crea un entorno virtual (recomendado):**
   ``` 
   python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3.  **Instala las dependencias:**
    (Asegúrate de haber creado un archivo `requirements.txt` con el comando `pip freeze > requirements.txt`)
    pip install -r requirements.txt

4.  **Ejecuta los notebooks:** Se recomienda empezar con el notebook de la práctica 1 para generar el archivo de datos limpio, y luego explorar el notebook de la carpeta `PIA/` para ver el análisis final.

## 📬 Contacto

**👤 Francisco Alexandro Gallegos Vidales**  
🔗 [GitHub](https://github.com/FrankLovelace)  
🌐 [Sitio Web](https://FrankLovelace.dev)  
📸 [Instagram](https://instagram.com/FrankLovegood)
