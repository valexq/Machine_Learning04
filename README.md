# Evento Evaluativo 4 - Análisis de Datos
# Machine Learning

Repositorio correspondiente al evento evaluativo 4 de la asignatura **Análisis de Datos**.  
En este proyecto se desarrollan dos ejercicios de Machine Learning asignados al grupo 5, aplicando análisis exploratorio de datos, preprocesamiento, entrenamiento de modelos, 
evaluación con métricas y visualización de resultados.

## Integrantes
## Grupo 5

| Nombre | Ejercicio | GitHub |
| --- | --- | --- |
| Vanessa Alfaro |  1 | `@valexq` |
| Ziuvar Ruiz |  1 | `@ziuvar` |
| Juan Manuel Valencia | 3 | `@Juanchos2905` |
| Juan Cardona | 3 | `@jcardser` |

## Ejercicios asignados

1. **Ejercicio 1:** Predicción del rendimiento académico de estudiantes.
2. **Ejercicio 3:** Análisis de sentimientos en reseñas de productos.

## Estructura del repositorio

```text
Machine_Learning04/
│   ├── ejercicio1/
│   │   ├── README.md
│   │   ├── notebooks/
│   │   └── data/
│   └── ejercicio3/
│       ├── README.md
│       ├── notebooks/
│       └── data/
├─ README.md
├─ .gitignore
└── requirements.txt

```

## Datasets utilizados

### Ejercicio 1 - Student Grade Prediction
Dataset orientado a la predicción del rendimiento académico de estudiantes a partir de variables demográficas, sociales y escolares.  
Fuente: [Kaggle - Student Grade Prediction](https://www.kaggle.com/datasets/dipam7/student-grade-prediction)

### Ejercicio 3 - Amazon Reviews Dataset
Dataset de reseñas de productos utilizado para clasificación de sentimientos en texto.  
Fuente: [Kaggle - Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

## Metodología

### Ejercicio 1 - Predicción del rendimiento académico

Se realizó un análisis exploratorio de datos para identificar distribuciones, correlaciones y posibles patrones entre variables académicas y personales.  
Posteriormente, se aplicó un proceso de preprocesamiento que incluyó limpieza de datos, imputación de valores faltantes, codificación de variables categóricas y estandarización de variables numéricas.

Los modelos evaluados fueron:

- Regresión Logística
- K-Nearest Neighbors (KNN)
- SVM

Las métricas utilizadas fueron:

- Accuracy
- Precision
- Recall
- F1-score

Además, se aplicó una técnica de reducción de dimensionalidad como PCA o t-SNE para visualizar la separabilidad entre clases.

### Ejercicio 3 - Análisis de sentimientos en reseñas

Se realizó limpieza y preprocesamiento del texto, incluyendo normalización, eliminación de stopwords, tokenización y vectorización mediante TF-IDF.  
Luego se entrenaron modelos de clasificación supervisada para identificar reseñas positivas y negativas.

Los modelos evaluados fueron:

- Naive Bayes
- Logistic Regression
- SVM

Las métricas utilizadas fueron:

- Accuracy
- F1-score
- Matriz de confusión

También se aplicó t-SNE para visualizar agrupamientos de reseñas en el espacio transformado.

## Resultados principales

### Ejercicio 1
- Modelo con mejor desempeño: Regresión Logística
- Métrica principal obtenida: Accuracy 0.86 y F1-score 0.86
- Hallazgo relevante: las notas previas `G1` y `G2` fueron las señales más fuertes para predecir el rendimiento final.

### Ejercicio 3
- Pendiente de actualización por los responsables del ejercicio 3.

## Cómo ejecutar el proyecto

1. Clonar el repositorio:
```bash
git clone https://github.com/valexq/Machine_Learning04.git
```

2. Ingresar a la carpeta del proyecto:
```bash
cd Machine_Learning04
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar los notebooks correspondientes desde la carpeta `notebooks/` de cada ejercicio.

## Tecnologías utilizadas

- Python
- Jupyter Notebook / Google Colab
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk / spaCy

## Video de sustentación

https://drive.google.com/file/d/1zJ_kqdZ4Yf_JgDoZJ2I44iYj69fhlt68/view?usp=sharing
