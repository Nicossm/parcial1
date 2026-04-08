# Proyecto Parcial 1: Análisis de Salarios de Desarrolladores de Software

## Integrantes
- [Nombre 1]
- [Nombre 2]
- [Nombre 3] (opcional)

## Descripción del Proyecto

Este proyecto forma parte de la Evaluación Parcial N°1 de la asignatura **Programación para la Ciencia de Datos (SCY1101)**. El objetivo es aplicar técnicas avanzadas de manipulación, limpieza y transformación de datos, generando insights y preparando la información para etapas posteriores de modelado.

## Dataset

### Origen
- **Fuente**: Kaggle - Software Developer Salary Dataset
- **Tema**: Salarios de desarrolladores de software a nivel mundial
- **Variable objetivo**: `salary_usd` (salario anual en USD)

### Descripción de las Variables

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `experience` | Numérico | Años de experiencia profesional |
| `country` | Categórico | País de residencia |
| `education` | Categórico | Nivel educativo más alto |
| `languages` | Texto | Lenguajes de programación principales |
| `frameworks` | Texto | Frameworks utilizados |
| `company_size` | Categórico | Tamaño de la empresa |
| `salary_usd` | Numérico | Salario anual en USD (variable objetivo) |

### Estado del Dataset (Raw)
El dataset "crudo" contiene las siguientes impurezas que serán tratadas:
- **Valores nulos**: ~7-8% en varias columnas
- **Duplicados**: ~200 filas duplicadas
- **Outliers**: Valores atípicos en salario y experiencia
- **Inconsistencias de formato**: Variaciones en nombres de países y niveles educativos
- **Tipos mezclados**: Valores de texto en columnas numéricas

## Estructura del Proyecto

```
parcial1/
├── data/
│   ├── raw/                    # Dataset original (sucio)
│   └── processed/              # Dataset limpio (resultado)
├── notebooks/
│   └── analisis_principal.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py        # Funciones de limpieza
│   ├── transformers.py         # Transformadores custom
│   └── utils.py                # Utilidades generales
├── outputs/
│   └── (gráficos exportados)
├── docs/
│   └── informe_tecnico.pdf
├── PLAN_PARCIAL.md
└── README.md
```

## Entorno de Desarrollo

### Justificación
[Elegir una opción y completar]

**Opción A: Google Colab**
- Ventaja: No requiere instalación local, fácil colaboración en tiempo real
- Integración con GitHub: Mediante commits directos desde Colab o descarga/push manual

**Opción B: Jupyter Local**
- Ventaja: Mayor control del entorno, sin dependencia de internet
- Integración con GitHub: Mediante git desde terminal

### Requisitos
```
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Cómo Ejecutar

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd parcial1
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar el notebook:
```bash
jupyter notebook notebooks/analisis_principal.ipynb
```

## Metodología

### 1. Análisis Exploratorio (EDA)
- Estadísticas descriptivas
- Detección de valores nulos y duplicados
- Identificación de outliers
- Visualizaciones iniciales

### 2. Limpieza de Datos
- Tratamiento de valores nulos (imputación/eliminación)
- Eliminación de duplicados
- Corrección de outliers
- Normalización de strings

### 3. Transformación
- Escalado de variables numéricas (StandardScaler/MinMaxScaler)
- Codificación de variables categóricas (OneHotEncoder/LabelEncoder)
- Pipeline de transformación con scikit-learn

### 4. Feature Engineering
- Creación de nuevas variables derivadas
- Categorización de variables continuas

## Resultados
[Completar después del análisis]

## Conclusiones
[Completar después del análisis]

## Licencia
Proyecto académico - Universidad [Nombre]
