# Plan de Trabajo - Parcial 1: Ciencia de Datos

## Información del Proyecto

- **Asignatura**: Programación para la Ciencia de Datos (SCY1101)
- **Evaluación**: Parcial N°1 (30% total: 10% Encargo Grupal + 20% Presentación Individual)
- **Dataset seleccionado**: `test.csv` (Software Developer Salary - ~10,000 filas)

---

## Fase 1: Ensuciar Dataset

### Objetivo
Transformar el dataset limpio `test.csv` en una versión "cruda" (`raw`) con impurezas realistas para poder demostrar técnicas de limpieza y transformación.

### Nivel de Suciedad: Moderado (~15-20% de datos afectados)

| Tipo de Impureza | Cantidad Aprox. | Columnas Afectadas | Descripción |
|------------------|-----------------|-------------------|-------------|
| **Valores NaN** | ~800 celdas | `education`, `salary_usd`, `experience`, `frameworks` | Celdas vacías/faltantes distribuidas aleatoriamente |
| **Duplicados** | ~300 filas | Todas las columnas | Filas repetidas exactas |
| **Outliers** | ~200 valores | `salary_usd`, `experience` | Salarios negativos o extremos (>$500k), experiencia negativa o >50 años |
| **Inconsistencias de formato** | ~500 valores | `country`, `education` | Variaciones: `"USA"` vs `"usa"` vs `"United States"`, espacios extra |
| **Tipos mezclados** | ~200 valores | `experience`, `salary_usd` | Texto en campos numéricos: `"N/A"`, `"unknown"`, `"-"` |

### Resultado Esperado
- Archivo: `data/raw/software_developer_salary_raw.csv`
- ~10,300 filas (incluyendo duplicados)
- Múltiples problemas de calidad de datos para limpiar

---

## Fase 2: Estructura del Proyecto

### Estructura de Carpetas Requerida

```
parcial1/
├── data/
│   ├── raw/
│   │   └── software_developer_salary_raw.csv    # Dataset sucio (original)
│   └── processed/
│       └── software_developer_salary_clean.csv  # Dataset limpio (resultado)
├── notebooks/
│   └── analisis_principal.ipynb                 # Notebook principal con EDA, limpieza, transformación
├── src/
│   ├── __init__.py                              # Hace el directorio un paquete Python
│   ├── data_cleaning.py                         # Funciones de limpieza de datos
│   ├── transformers.py                          # Transformadores custom para pipelines
│   └── utils.py                                 # Funciones utilitarias generales
├── outputs/
│   └── (gráficos y visualizaciones exportadas)
├── docs/
│   └── informe_tecnico.pdf                      # Informe técnico (8-12 páginas)
├── README.md                                    # Documentación del proyecto
└── PLAN_PARCIAL.md                              # Este archivo
```

### Contenido del README.md
- Nombre de los integrantes
- Origen y descripción de los datos
- Justificación del entorno (Google Colab vs Jupyter Local)
- Instrucciones de reproducibilidad

---

## Fase 3: Desarrollo del Notebook (Pendiente)

### Secciones del Notebook

1. **Análisis Exploratorio Inicial (EDA)**
   - Estadísticas descriptivas (`describe()`, `info()`)
   - Detección de nulos (`isnull().sum()`)
   - Detección de duplicados (`duplicated().sum()`)
   - Detección de outliers (IQR, boxplots)
   - Visualizaciones iniciales (distribuciones, correlaciones)

2. **Pipeline de Limpieza**
   - Manejo de NaN (imputación con media/mediana/moda o eliminación)
   - Eliminación de duplicados
   - Tratamiento de outliers (IQR, Z-score, capping)
   - Normalización de strings (lowercase, strip, mapeo de variantes)
   - Conversión de tipos de datos

3. **Pipeline de Transformación**
   - `StandardScaler` / `MinMaxScaler` para variables numéricas
   - `OneHotEncoder` / `LabelEncoder` para variables categóricas
   - Uso de `sklearn.pipeline.Pipeline` y `ColumnTransformer`

4. **Feature Engineering**
   - `salary_per_year_experience` = salary_usd / experience
   - `is_senior` = experience > 10 (booleano)
   - `country_tier` = categorización de países por nivel de salario
   - `num_languages` = contar cantidad de lenguajes
   - `num_frameworks` = contar cantidad de frameworks
   - `experience_category` = Junior/Mid/Senior/Lead

5. **Validación y Resultados**
   - Comparación antes/después (shape, nulls, dtypes)
   - Métricas de calidad de datos
   - Visualizaciones finales

---

## Fase 4: Módulos en `/src/` (Pendiente)

### data_cleaning.py
```python
def remove_duplicates(df): ...
def handle_missing_values(df, strategy='mean'): ...
def fix_outliers(df, column, method='iqr'): ...
def normalize_strings(df, column): ...
```

### transformers.py
```python
class CustomScaler: ...
class CategoryEncoder: ...
```

### utils.py
```python
def load_data(path): ...
def save_data(df, path): ...
def get_data_quality_report(df): ...
```

---

## Requisitos Técnicos del Parcial

### Código
- [ ] Código limpio y bien comentado
- [ ] Docstrings en funciones
- [ ] Reutilización de funciones (modularidad)
- [ ] Manejo de excepciones
- [ ] Uso de: Python, Jupyter, Pandas, NumPy, Scikit-learn

### Documentación
- [ ] Markdown explicativo en cada sección del notebook
- [ ] Justificación técnica de cada decisión
- [ ] README.md completo

### Versionado
- [ ] Repositorio en GitHub
- [ ] Commits de todos los integrantes
- [ ] Código reproducible ("corre a la primera")

### Informe Técnico (8-12 páginas)
- [ ] Resumen ejecutivo
- [ ] Análisis exploratorio inicial
- [ ] Metodología de transformación
- [ ] Resultados y validación
- [ ] Conclusiones y recomendaciones

---

## Notas Adicionales

- **Fuente original del dataset**: Kaggle - Software Developer Salary
- **Columnas disponibles**: `experience`, `country`, `education`, `languages`, `frameworks`, `company_size`, `salary_usd`
- **Variable objetivo**: `salary_usd` (para futuro modelado predictivo)
