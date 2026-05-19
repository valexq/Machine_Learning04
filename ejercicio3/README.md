# Ejercicio 3: Análisis de Sentimientos en Reseñas de Productos

## Descripción General

Este ejercicio desarrolla un **modelo de aprendizaje supervisado** para clasificar reseñas de productos como positivas o negativas. Se implementa una pipeline completa de procesamiento de texto natural (NLP), desde la limpieza y preprocesamiento hasta el entrenamiento y evaluación de múltiples algoritmos de clasificación.

**Dataset:** Amazon Reviews Dataset en formato FastText (.ft.txt)  
**Objetivo:** Clasificar reseñas de productos mediante análisis de sentimientos

---

## Requerimientos

1. Limpieza y preprocesamiento del texto: (stopwords, tokenización, TF-IDF)
2. Entrenar modelos de clasificación: Naive Bayes, Logistic Regression y SVM
3. Evaluar rendimiento: Accuracy, F1-score y Matriz de Confusión
4. Reducción de dimensionalidad: t-SNE para visualizar agrupaciones

---

## Cómo ejecutar

1. Clonar el repositorio.
2. Instalar las dependencias necesarias.
3. Ejecutar el notebook principal `notebooks/Ejercicio3.ipynb`.


## 1. Dataset

### Formato y Características

El dataset está en formato **FastText** (`.ft.txt`), donde cada línea contiene una etiqueta y el texto de una reseña:


__label__1 texto de reseña negativa aquí...
__label__2 texto de reseña positiva aquí...


- __label__1 = Reseña **negativa** (1-2 estrellas)
- __label__2 = Reseña **positiva** (4-5 estrellas)

### Parámetros de Carga

- **RUTA_DATASET**: `../data/train.ft.txt`
- **TAMANO_MUESTRA**: 200,000 registros (para reducir tiempo de procesamiento)
- **Total de registros en archivo**: > 3,000,000 reseñas

### Tabla de Datos (DataFrame)

Después del parsing, los datos se organizan en un DataFrame con las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `review` | str | Texto original de la reseña |
| `sentiment` | int | Etiqueta binaria: 0=Negativo, 1=Positivo |
| `clean_review` | str | Texto después del preprocesamiento |

---

## 2. Preprocesamiento del Texto

Se implementó una **pipeline de limpieza** con los siguientes pasos:

### 2.1 Función `limpiar_texto()`

```python
def limpiar_texto(texto: str) -> str:
    1. Convertir a minúsculas (lowercase)
    2. Eliminar caracteres especiales (números, puntuación, símbolos)
    3. Tokenización con NLTK (word_tokenize)
    4. Eliminar stopwords del inglés (palabras sin valor semántico: "the", "is", "a", etc.)
    5. Filtrar tokens de menos de 2 caracteres
    6. Reconstruir texto limpio
```

### 2.2 Vectorización con TF-IDF

Después de la limpieza, se convierte el texto a **representaciones numéricas** usando TF-IDF:

```python
TfidfVectorizer(
    max_features=5000,         
    ngram_range=(1, 2),        
    sublinear_tf=True          
)
```

**Resultado:** Matriz de 200,000 documentos × 5,000 características

---

## 3. División de Datos

```
Conjunto completo (200,000 muestras)
    ├─ Entrenamiento (80%): 160,000 muestras → X_train, y_train
    └─ Prueba (20%):         40,000 muestras → X_test, y_test

Estratificación: stratify=y para mantener la misma proporción 
                 de clases positivas/negativas en ambos conjuntos
```

---

## 4. Modelos de Clasificación

Se entrenaron y evaluaron tres algoritmos clásicos de aprendizaje supervisado:

### 4.1 Naive Bayes (MultinomialNB)

- **Descripción:** Modelo probabilístico basado en la independencia condicional de palabras
- **Ventajas:** Muy rápido, excelente baseline, eficiente con datos de alta dimensionalidad
- **Parámetros:** `alpha=0.1` (suavizado Laplace)

### 4.2 Logistic Regression

- **Descripción:** Modelo lineal que aprende pesos para cada característica (palabra)
- **Ventajas:** Interpretable, eficiente, funciona bien en datos de alta dimensionalidad
- **Parámetros:** 
  - `max_iter=1000`
  - `C=1.0` (inverso de fuerza de regularización)
  - `solver='lbfgs'`

### 4.3 SVM (LinearSVC)

- **Descripción:** Busca el hiperplano de máximo margen entre clases
- **Ventajas:** Muy robusto en NLP, suele tener el mejor rendimiento en texto
- **Parámetros:**
  - `C=1.0` (parámetro de penalización)
  - `max_iter=2000`
  - `dual=False` (mejor para n_samples > n_features)

---

## 5. Métricas de Evaluación

Cada modelo se evaluó con las siguientes métricas en el conjunto de prueba (40,000 muestras):

### 5.1 Accuracy

$$\text{Accuracy} = \frac{\text{Predicciones correctas}}{\text{Total de muestras}}$$

- Proporción general de predicciones correctas
- Rango: [0, 1] (0=peor, 1=perfecto)

### 5.2 F1-Score (ponderado)

$$F1\text{-}Score = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}$$

- Media armónica entre Precisión (¿cuántas reseñas positivas predichas eran realmente positivas?) y Recall (¿cuántas reseñas positivas reales fueron encontradas?)
- Ponderado: ajusta según la proporción de clases
- Rango: [0, 1]

### 5.3 Matriz de Confusión

Visualiza los cuatro tipos de predicciones:

```
                    Predicción
                Negativo  Positivo
Real  Negativo    TN        FP
      Positivo    FN        TP

TN = Verdaderos Negativos (reseña negativa predicha como negativa)
FP = Falsos Positivos (reseña negativa predicha como positiva)
FN = Falsos Negativos (reseña positiva predicha como negativa)
TP = Verdaderos Positivos (reseña positiva predicha como positiva)
```

### 5.4 Reporte de Clasificación

Se generó un `classification_report()` para cada modelo con métricas por clase:
- **Precision:** De todas las predicciones positivas, ¿cuántas eran correctas?
- **Recall:** De todas las muestras positivas reales, ¿cuántas se detectaron?
- **F1-Score:** Media armónica de Precision y Recall
- **Support:** Número total de muestras de cada clase

---

## 6. Reducción de Dimensionalidad con t-SNE

### 6.1 ¿Qué es t-SNE?

**t-SNE** (t-distributed Stochastic Neighbor Embedding) es una técnica de reducción de dimensionalidad **no lineal** que proyecta datos de alta dimensión (5,000 características en nuestro caso) a 2D preservando las relaciones locales entre puntos.

### 6.2 Aplicación

```python
TSNE(
    n_components=2,          # Reducir a 2 dimensiones para visualización
    perplexity=30,          # Balancea atención local vs global
    max_iter=1000,          # Iteraciones de optimización
    learning_rate='auto',   # Tasa de aprendizaje automática
    init='pca'              # Inicialización con PCA para mayor estabilidad
)
```

### 6.3 Muestra para Visualización

- **Tamaño:** 1,000 documentos (50% positivos, 50% negativos)
- **Justificación:** t-SNE es computacionalmente costoso; 1,000 muestras permiten visualización legible sin tiempos prohibitivos
- **Estratificación:** Se mantiene la proporción de clases

### 6.4 Interpretación del Gráfico

El gráfico t-SNE muestra:
- **Eje X e Y:** Componentes t-SNE (sin interpretación semántica directa)
- **Color Rojo:** Reseñas negativas
- **Color Verde:** Reseñas positivas
- **Clusters:** Si hay separación clara entre colores, indica que los vectores TF-IDF distinguen bien entre sentimientos

---

## 7. Función de Evaluación

Se encapsuló el proceso de entrenamiento y evaluación en `evaluar_modelo()`:

```python
def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_te, y_te, color_mapa='Blues'):
    1. Entrenar el modelo con X_train, y_train
    2. Realizar predicciones en X_test
    3. Calcular Accuracy y F1-Score ponderado
    4. Mostrar reporte de clasificación detallado
    5. Visualizar matriz de confusión con color personalizado
    6. Retornar diccionario con resultados y objeto del modelo
```

**Ventaja:** Evita código repetido y facilita comparaciones equitativas entre modelos.

---

## 8. Comparación de Modelos

Los tres modelos se compararon en términos de:
- **Accuracy:** Precisión general en el conjunto de prueba
- **F1-Score ponderado:** Métrica balanceada considerando desbalance de clases

Se generó un gráfico de barras mostrando lado a lado la comparación.

---

## 9. Variables Clave

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `df` | DataFrame | Dataset completo con columnas: `review`, `sentiment`, `clean_review` |
| `X` | scipy.sparse matrix | Matriz TF-IDF de 200,000 × 5,000 (representación numérica del texto) |
| `y` | Series | Vector de etiquetas binarias (0=Negativo, 1=Positivo) |
| `X_train` | scipy.sparse matrix | 160,000 × 5,000 - datos de entrenamiento |
| `X_test` | scipy.sparse matrix | 40,000 × 5,000 - datos de prueba |
| `y_train` | Series | 160,000 etiquetas de entrenamiento |
| `y_test` | Series | 40,000 etiquetas de prueba |
| `vectorizer` | TfidfVectorizer | Objeto que transforma texto a vectores TF-IDF |
| `stop_words` | set | Palabras en inglés sin valor semántico (1,179 palabras) |
| `X_sample` | ndarray | 1,000 × 5,000 - muestra para t-SNE (convertida a denso) |
| `X_tsne` | ndarray | 1,000 × 2 - proyección 2D de t-SNE |
| `resultados` | DataFrame | Tabla con métricas de los tres modelos |
| `res_nb`, `res_lr`, `res_svm` | dict | Diccionarios con resultados de cada modelo |

---

## 10. Resultados Experimentales

### 10.1 Rendimiento de los Modelos

Los tres modelos fueron entrenados con 160,000 muestras y evaluados en 40,000 muestras de prueba. Los resultados fueron los siguientes:

| Modelo | Accuracy | F1-Score | Ranking |
|--------|----------|----------|---------|
| **Logistic Regression** | **0.8915** | **0.8914** | 🥇 1º lugar |
| SVM (LinearSVC) | 0.8903 | 0.8903 | 🥈 2º lugar |
| Naive Bayes | 0.8657 | 0.8657 | 🥉 3º lugar |

### 10.2 Análisis por Modelo

#### Logistic Regression (Ganador)

- **Accuracy:** 89.15% - El modelo clasifica correctamente 89 de cada 100 reseñas
- **Precisión por clase:**
  - Negativo: 89% de las reseñas predichas como negativas eran correctas
  - Positivo: 89% de las reseñas predichas como positivas eran correctas
- **Recall por clase:**
  - Negativo: Detectó el 88% de las reseñas negativas reales
  - Positivo: Detectó el 90% de las reseñas positivas reales
- **Matriz de Confusión:**
  - Verdaderos Negativos: 17,481
  - Verdaderos Positivos: 18,177
  - Falsos Positivos: 2,286
  - Falsos Negativos: 2,056

#### SVM (LinearSVC)

- **Accuracy:** 89.03% - Solo 0.12 puntos porcentuales por debajo de Logistic Regression
- **Performance casi idéntica** a Logistic Regression, con ligeras variaciones en la clasificación
- **Matriz de Confusión:**
  - Verdaderos Negativos: 17,502
  - Verdaderos Positivos: 18,110
  - Falsos Positivos: 2,265
  - Falsos Negativos: 2,123

#### Naive Bayes

- **Accuracy:** 86.57% - Desempeño inferior a los otros dos modelos
- **Diferencia:** 2.58 puntos porcentuales por debajo de Logistic Regression
- **Matriz de Confusión:**
  - Verdaderos Negativos: 17,003
  - Verdaderos Positivos: 17,624
  - Falsos Positivos: 2,764
  - Falsos Negativos: 2,609
- **Interpretación:** A pesar de su desempeño más bajo, el modelo sigue siendo útil y muy rápido de entrenar

### 10.3 Balanceo del Dataset

El dataset utilizado está **perfectamente balanceado:**
- Reseñas negativas: 98,834 (49.4%)
- Reseñas positivas: 101,166 (50.6%)

**Implicación:** El balanceo ideal significa que el Accuracy es una métrica confiable, sin necesidad de aplicar técnicas de balanceo de clases.

### 10.4 Visualización t-SNE

La proyección 2D de t-SNE revela:

- **Patrón observado:** Hay una distribución aproximadamente uniforme de los puntos en el espacio 2D
- **Mezcla de colores:** Se observa tanto clustering parcial como solapamiento considerable entre reseñas positivas (verde) y negativas (rojo)
- **Interpretación:**
  - No hay separación perfecta entre clases en el espacio TF-IDF
  - Los vectores de palabras contienen información útil pero no completamente discriminativa
  - La existencia de solapamiento explica por qué los modelos tienen accuracy de ~89% y no cercana al 100%
  - Algunas reseñas tienen características lingüísticas similares independientemente del sentimiento

### 10.5 Análisis de Errores

**Tasa de error promedio:** ~11%

**Causas potenciales de clasificación incorrecta:**
1. **Lenguaje ambiguo:** Reseñas con crítica constructiva ("no es malo pero podría mejorar")
2. **Sarcasmo:** Comentarios sarcásticos donde se dice lo opuesto a lo que se siente
3. **Palabras mixtas:** Reseñas que contienen tanto aspectos positivos como negativos
4. **Datos etiquetados inconsistentemente:** Posibles errores en el etiquetado original del dataset

### 10.6 Impacto de los Bigramas

La inclusión de bigramas (`ngram_range=(1,2)`) en la vectorización TF-IDF fue crucial:

- **Unigramas solos:** Capturaría "good", "bad", "not" como palabras individuales
- **Bigramas:** Captura "not good", "very bad", "really loved", que tienen significado semántico completo
- **Resultado:** Mejora la discriminación entre sentimientos, evitando confusiones por palabras negativas mal interpretadas

---

## 11. Conclusiones

### Hallazgos Principales

**1. Rendimiento General:**  
Los modelos logran un Accuracy de **~89%**, demostrando que el análisis de sentimientos en reseñas de Amazon es una tarea bien resuelta con métodos clásicos de Machine Learning.

**2. Superioridad de Logistic Regression:**  
Entre los tres modelos probados, **Logistic Regression obtuvo el mejor rendimiento** (89.15%), aunque SVM fue muy cercano (89.03%). Naive Bayes, aunque más rápido, fue menos preciso (86.57%).

**3. Preprocesamiento Efectivo:**  
La pipeline de limpieza de texto fue fundamental para alcanzar estos resultados:
- Conversión a minúsculas: Normaliza variaciones tipográficas
- Eliminación de caracteres especiales: Reduce ruido
- Tokenización y remoción de stopwords: Elimina palabras sin valor semántico
- TF-IDF con bigramas: Captura expresiones semánticas completas

**4. Dataset Perfectamente Balanceado:**  
La proporción 50-50 de reseñas positivas y negativas garantiza que Accuracy sea una métrica confiable sin sesgo hacia una clase.

**5. Visualización t-SNE Interpretable:**  
La proyección 2D muestra que aunque hay cierto grado de separación entre sentimientos, existe un área de solapamiento considerable. Esto explica:
- Por qué los modelos no alcanzan perfección (100%)
- Por qué el error del ~11% es esperable
- La necesidad de otros contextos (usuario, categoría de producto) para mejorar las predicciones

**6. Matriz de Confusión Balanceada:**  
Los falsos positivos y falsos negativos están relativamente balanceados, indicando que los modelos no tienen sesgo hacia ninguna clase.

### Recomendaciones Futuras

1. **Modelos Avanzados:**
   - Experimentar con LSTM, GRU para capturar dependencias contextuales
   - Usar transformers preentrenados (BERT, DistilBERT, RoBERTa) que cumplen con 90%+ de accuracy
   - Fine-tuning de modelos preentrenados específicos para análisis de sentimientos

2. **Optimización de Hiperparámetros:**
   - GridSearchCV o RandomizedSearchCV para ajustar C, regularization, learning_rate
   - Validación cruzada (k-fold) para mayor robustez en las métricas

3. **Análisis de Características:**
   - Identificar palabras más influyentes por clase usando `feature_importances_` o LIME
   - Crear visualizaciones de palabras clave para cada sentimiento

4. **Manejo de Ambigüedad:**
   - Crear categoría "Neutral" para reseñas mixtas (si el dataset lo permite)
   - Usar confianza de predicción para filtrar casos ambiguos (predicciones cercanas a 0.5)
   - Implementar métodos de explicabilidad (LIME, SHAP) para casos de error

5. **Ensamble de Modelos:**
   - Combinar predicciones de los tres modelos (votación mayoritaria)
   - Usar stacking para aprovechar fortalezas de cada modelo

6. **Análisis Contextual:**
   - Incorporar metadatos: categoría de producto, número de estrellas, fecha de reseña
   - Analizar reseñas por categoría de producto (ropa vs electrónica vs libros)

7. **Validación Real:**
   - Evaluar el modelo con reseñas nuevas en tiempo real
   - Implementar feedback loop para reentrenamiento periódico
   - Monitorear drift de datos (cambios en patrones de reseñas)

---

## 12. Archivos del Proyecto

```
ejercicio3/
├── README.md                               (este archivo)
├── data/
│   ├── train.ft.txt                        (dataset: 3M+ reseñas en formato FastText)
│   └── test.ft.txt                         (dataset de prueba)
└── notebooks/
    └── Ejercicio3.ipynb                    (notebook con el análisis completo)
```

---

## 13. Librerías Utilizadas

- **pandas, numpy**: Manipulación de datos y cálculos numéricos
- **scikit-learn**: Algoritmos de ML, vectorización TF-IDF, métricas
- **nltk**: Tokenización, stopwords, procesamiento de lenguaje natural
- **matplotlib, seaborn**: Visualización de gráficos y matrices
- **re**: Expresiones regulares para limpieza de texto

