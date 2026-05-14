# Ejercicio 1 - Predicción del rendimiento académico de estudiantes

## Información general

Este ejercicio corresponde al Evento Evaluativo 4 de la asignatura **Análisis de Datos**. 
El objetivo consiste en construir un modelo de clasificación multiclase para predecir el nivel 
de rendimiento académico de estudiantes a partir de variables demográficas, familiares, sociales y académicas.

El problema se formuló como una tarea de **clasificación supervisada**, donde la variable objetivo `rendimiento`
se construyó a partir de la nota final `G3`, agrupando a los estudiantes en tres categorías: **bajo**, **medio** 
y **alto**. Esta transformación permite convertir la predicción del desempeño en un problema de clasificación 
multiclase interpretable y útil para fines de alerta temprana.

## Motivación

La predicción del rendimiento académico tiene valor porque permite identificar estudiantes con
mayor riesgo de bajo desempeño y apoyar procesos de acompañamiento, intervención temprana y toma 
de decisiones educativas.

Además de su utilidad predictiva, este tipo de análisis permite estudiar cómo variables como 
el tiempo de estudio, las ausencias, los antecedentes académicos y el entorno familiar se relacionan
con los resultados académicos, aportando evidencia para comprender mejor los factores asociados al rendimiento.

## Objetivos

### Objetivo general

Desarrollar y comparar modelos de Machine Learning para predecir el nivel de rendimiento académico de
estudiantes a partir de variables académicas, personales y familiares.

### Objetivos específicos

- Realizar un análisis exploratorio de datos para identificar distribuciones, correlaciones y patrones por subgrupos relevantes.
- Construir una variable objetivo categórica de rendimiento a partir de la nota final `G3`.
- Implementar un preprocesamiento completo que incluya imputación, estandarización y codificación de variables.
- Entrenar y comparar modelos de clasificación supervisada como Regresión Logística, KNN y un tercer modelo adicional.
- Evaluar el desempeño de los modelos mediante métricas como Accuracy, Precision, Recall y F1-score.
- Interpretar los resultados obtenidos y discutir sus implicaciones éticas en contextos educativos.

## Dataset

Se utilizó el dataset **Student Performance**, ampliamente empleado en estudios de analítica educativa. 
Este conjunto de datos contiene información de estudiantes de educación secundaria en Portugal e incluye 
variables relacionadas con el colegio, sexo, edad, condiciones familiares, apoyo académico, hábitos y
calificaciones parciales y finales.

Entre las variables más relevantes del dataset se encuentran:

- `age`: edad del estudiante.
- `studytime`: tiempo de estudio semanal.
- `failures`: número de asignaturas reprobadas previamente.
- `absences`: número de ausencias.
- `G1`, `G2`, `G3`: notas del primer periodo, segundo periodo y nota final, respectivamente.
- `famsup`, `schoolsup`, `Pstatus`: variables relacionadas con apoyo familiar, apoyo escolar y convivencia de los padres.

La variable objetivo original no existe de forma categórica en el dataset. Por ello, se construyó la variable 
`rendimiento` a partir de `G3`, agrupando las notas finales en tres niveles:

- `bajo`: 0 a 9.
- `medio`: 10 a 14.
- `alto`: 15 a 20.

## Metodología

### Exploración de datos

Se realizó una exploración general del dataset mediante revisión de dimensiones, tipos de variables, 
estadísticas descriptivas y verificación de valores faltantes. Posteriormente, se analizaron distribuciones 
de la nota final `G3`, de la variable objetivo `rendimiento`, de la edad y de las ausencias por nivel de rendimiento.

También se construyó una matriz de correlación para las variables numéricas, observando relaciones fuertes entre `G3` 
y las notas previas `G1` y `G2`, así como relaciones negativas entre el rendimiento final y variables como `failures` y `absences`.

Finalmente, se realizaron análisis de subgrupos para estudiar la distribución del rendimiento según variables
como sexo (`sex`), apoyo familiar (`famsup`) y estado de convivencia de los padres (`Pstatus`), con el fin de 
identificar posibles patrones diferenciales entre grupos.

### Preprocesamiento

Se separó la matriz de características `X` de la variable objetivo `rendimiento`. Las variables numéricas 
se procesaron mediante imputación por mediana y estandarización con `StandardScaler`, mientras que las 
categóricas se trataron con imputación por el valor más frecuente y codificación One-Hot mediante `OneHotEncoder`.

Todo el flujo de preparación de datos se integró en un `ColumnTransformer` dentro de un pipeline de scikit-learn,
garantizando una transformación consistente de los datos de entrenamiento y prueba y evitando fugas de información 
entre ambos conjuntos.

### Modelos implementados

#### Regresión Logística

La Regresión Logística se utilizó como modelo base de clasificación multiclase, ya que permite estimar
probabilidades de pertenencia a cada categoría de rendimiento y ofrece un alto nivel de interpretabilidad
sobre el efecto de las variables explicativas.

Su principal fortaleza en este contexto es que sirve como línea base clara y explicable para un problema 
tabular estructurado. Sin embargo, puede ser menos adecuada si las fronteras entre clases son altamente 
no lineales o si existen interacciones complejas entre variables.

#### K-Nearest Neighbors (KNN)

KNN clasifica cada estudiante a partir de la clase predominante entre sus vecinos más cercanos 
en el espacio de características, lo que resulta razonable cuando estudiantes con perfiles 
similares tienden a presentar niveles de rendimiento parecidos.

Este algoritmo puede capturar patrones locales no lineales, aunque depende fuertemente 
de la escala de las variables y puede verse afectado por la alta dimensionalidad generada 
por la codificación de variables categóricas.

#### Tercer modelo

En esta sección debe documentarse el tercer modelo utilizado en el ejercicio, ya sea **SVM** o 
**Random Forest**, junto con su justificación teórica, ventajas, limitaciones y resultados obtenidos.

## Métricas de evaluación

Los modelos se evaluaron mediante **Accuracy**, **Precision**, **Recall** y **F1-score**, 
métricas adecuadas para comparar clasificadores multiclase cuando interesa valorar tanto el
porcentaje global de aciertos como el equilibrio del desempeño entre categorías.

### Resultados actuales

| Modelo | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Regresión Logística | 0.86 | 0.86 | 0.86 | 0.86 |
| KNN | Completar | Completar | Completar | Completar |
| Tercer modelo | Completar | Completar | Completar | Completar |

### Detalle por clase - Regresión Logística

| Clase | Precision | Recall | F1-score | Soporte |
|------|-----------|--------|----------|---------|
| alto | 0.87 | 0.87 | 0.87 | 15 |
| bajo | 0.83 | 0.92 | 0.87 | 26 |
| medio | 0.89 | 0.82 | 0.85 | 38 |

Estos resultados muestran un desempeño global sólido de la Regresión Logística, 
con un Accuracy de 0.86 y un equilibrio razonable entre precisión y sensibilidad en las tres clases.
El modelo identifica especialmente bien la clase `bajo` en términos de recall, 
mientras que en la clase `medio` presenta un recall ligeramente menor, lo que sugiere una mayor
dificultad para distinguir algunos casos intermedios.

## Visualización con PCA

Se aplicó PCA para proyectar los estudiantes en un espacio bidimensional y visualizar la 
distribución de las clases de rendimiento. Esta técnica de reducción de dimensionalidad 
permite observar de manera global si existe separabilidad entre categorías y qué tan solapadas 
se encuentran en el espacio de características transformado.

En la visualización se aprecia un solapamiento importante entre las clases `bajo`, `medio` y `alto`, 
aunque también se identifican zonas donde los estudiantes de rendimiento alto tienden a concentrarse 
hacia valores positivos del primer componente principal. Esto sugiere que existen patrones útiles 
para la clasificación, pero no una separación perfecta entre clases, lo cual ayuda a explicar por qué 
los modelos no alcanzan un rendimiento ideal.

## Interpretación de resultados

Los resultados obtenidos indican que el rendimiento académico puede predecirse con una precisión
razonable utilizando variables académicas y de contexto. En particular, las notas previas (`G1` y `G2`)
aparecen como señales muy fuertes del desempeño final, mientras que variables como ausencias, tiempo de
estudio y fallos previos aportan información adicional relevante.

La comparación entre modelos permite discutir si la estructura del problema parece más lineal 
o más local/no lineal. Si KNN o el tercer modelo superan claramente a la Regresión Logística, 
ello sugeriría que existen relaciones más complejas entre las variables; si el desempeño es similar, 
la Regresión Logística se consolida como una alternativa especialmente valiosa por su interpretabilidad.

## Implicaciones éticas

El uso de modelos de Machine Learning para predecir el rendimiento académico implica riesgos éticos relevantes. 
En primer lugar, el modelo puede aprender patrones que reflejan desigualdades sociales o familiares ya existentes
en los datos, reforzando sesgos relacionados con el contexto del estudiante en lugar de limitarse a medir su 
desempeño académico.

Además, clasificar a un estudiante como de bajo rendimiento potencial puede contribuir a su 
estigmatización si la predicción se interpreta como un juicio definitivo y no como una herramienta 
de apoyo. Estas predicciones no deberían emplearse para restringir oportunidades, sino para activar 
mecanismos de acompañamiento, tutoría y seguimiento pedagógico.

Por ello, cualquier aplicación real de este tipo de modelos debería incluir supervisión humana, 
transparencia en las variables utilizadas, revisión periódica de sesgos y protección rigurosa de los
datos personales y académicos de los estudiantes.

## Estructura sugerida del ejercicio

```text
Ejercicio1/
├── README.md
├── notebooks/
│   └── ejercicio1_rendimiento_estudiantes.ipynb
└── data/
    └── student-mat.csv
```

## Cómo ejecutar

1. Clonar el repositorio.
2. Instalar las dependencias necesarias.
3. Ejecutar el notebook principal del ejercicio 1.
4. Revisar las secciones de EDA, preprocesamiento, modelado, métricas, PCA y conclusiones.

Dependencias sugeridas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Conclusiones
