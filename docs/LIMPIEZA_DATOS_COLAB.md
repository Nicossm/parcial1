# 2. Limpieza de Datos

Este bloque está pensado para continuar tu notebook después de la sección de EDA y visualizaciones iniciales. Se asume que ya ejecutaste las importaciones, ya cargaste el archivo CSV y que el DataFrame original se llama `df`.

## 2.1 Objetivo de esta etapa

En esta sección se realizará la limpieza del dataset siguiendo un flujo ordenado y justificable:

- crear una copia de trabajo
- unificar nulos y pseudonulos
- normalizar variables categóricas
- convertir tipos de datos
- eliminar duplicados
- tratar valores faltantes
- corregir valores imposibles y outliers
- validar resultados y exportar el dataset limpio

## 2.2 Copia de trabajo para no modificar el DataFrame original

```python
# Copia de trabajo para no modificar el DataFrame original
df_clean = df.copy()

print("Shape original:", df.shape)
print("Shape copia de trabajo:", df_clean.shape)
```

---

## 2.3 Funciones auxiliares de limpieza

### 2.3.1 Función para normalizar listas separadas por coma

Esta función se usará en columnas como `languages` y `frameworks` para:

- quitar espacios extra
- eliminar elementos repetidos dentro de la celda
- ordenar alfabéticamente
- dejar una representación consistente

```python
def normalizar_lista_tecnologias(valor):
    if pd.isna(valor):
        return np.nan
    
    tokens = [x.strip() for x in str(valor).split(",") if x.strip() != ""]
    
    if len(tokens) == 0:
        return np.nan
    
    # Eliminar repetidos internos y ordenar para estandarizar
    tokens_unicos = sorted(set(tokens), key=str.lower)
    
    return ", ".join(tokens_unicos)
```

---

### 2.3.2 Función para normalizar países

```python
def normalizar_country(valor):
    if pd.isna(valor):
        return np.nan
    
    v = str(valor).strip().upper()
    
    mapa_paises = {
        "USA": "USA",
        "US": "USA",
        "UNITED STATES": "USA",
        "UNITED STATES OF AMERICA": "USA",
        "UK": "UK",
        "UNITED KINGDOM": "UK",
        "FR": "France",
        "FRANCE": "France",
        "GERMANY": "Germany",
        "JAPAN": "Japan",
        "INDIA": "India",
        "BRAZIL": "Brazil",
        "CANADA": "Canada",
        "AUSTRALIA": "Australia",
        "SINGAPORE": "Singapore"
    }
    
    if v in mapa_paises:
        return mapa_paises[v]
    
    # Si no está en el mapa, usar formato título
    return str(valor).strip().title()
```

---

### 2.3.3 Función para normalizar educación

```python
def normalizar_education(valor):
    if pd.isna(valor):
        return np.nan
    
    v = str(valor).strip().upper()
    
    mapa_education = {
        "HIGH SCHOOL": "High School",
        "SOME COLLEGE": "Some College",
        "BACHELORS": "Bachelors",
        "BACHELOR": "Bachelors",
        "MASTERS": "Masters",
        "MASTER": "Masters",
        "PHD": "PhD",
        "PHD.": "PhD"
    }
    
    return mapa_education.get(v, str(valor).strip().title())
```

---

### 2.3.4 Función para resumir outliers con IQR

```python
def resumen_outliers_iqr(df, columna):
    serie = pd.to_numeric(df[columna], errors="coerce").dropna()
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    mascara_outliers = (serie < limite_inferior) | (serie > limite_superior)
    cantidad_outliers = mascara_outliers.sum()
    
    return {
        "columna": columna,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "cantidad_outliers": int(cantidad_outliers)
    }
```

---

### 2.3.5 Función para aplicar capping con IQR

```python
def cap_outliers_iqr(serie):
    serie = pd.to_numeric(serie, errors="coerce")
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    serie_cap = serie.clip(lower=limite_inferior, upper=limite_superior)
    
    return serie_cap, limite_inferior, limite_superior
```

---

## 2.4 Unificación de nulos y pseudonulos

Antes de imputar o eliminar datos, primero hay que convertir todos los valores que significan “faltante” a un único formato (`NaN`).

Entre ellos se observaron casos como:

- `missing`
- `null`
- `NaN`
- `N/A`
- `n/a`
- `unknown`
- `?`
- celdas vacías

```python
# Respaldo del conteo inicial de nulos
nulos_antes = df_clean.isnull().sum()

# Lista de pseudonulos a unificar
pseudonulos = [
    "missing", "Missing",
    "null", "NULL",
    "NaN", "nan",
    "N/A", "n/a",
    "NA", "na",
    "unknown", "Unknown",
    "?", "-", ""
]

df_clean = df_clean.replace(pseudonulos, np.nan)

# Conteo posterior
nulos_despues_unificacion = df_clean.isnull().sum()

comparacion_nulos = pd.DataFrame({
    "nulos_antes": nulos_antes,
    "nulos_despues_unificacion": nulos_despues_unificacion,
    "incremento": nulos_despues_unificacion - nulos_antes
})

print("Comparación de nulos antes y después de unificar pseudonulos:")
display(comparacion_nulos)
```

---

## 2.5 Normalización de variables categóricas

### 2.5.1 Normalizar `country` y `education`

```python
df_clean["country"] = df_clean["country"].apply(normalizar_country)
df_clean["education"] = df_clean["education"].apply(normalizar_education)

print("Valores únicos de country después de normalizar:")
print(sorted(df_clean["country"].dropna().unique())[:20])

print("\nValores únicos de education después de normalizar:")
print(sorted(df_clean["education"].dropna().unique()))
```

---

### 2.5.2 Normalizar `languages` y `frameworks`

```python
df_clean["languages"] = df_clean["languages"].apply(normalizar_lista_tecnologias)
df_clean["frameworks"] = df_clean["frameworks"].apply(normalizar_lista_tecnologias)

print("Ejemplo de values_counts en languages:")
display(df_clean["languages"].value_counts(dropna=False).head(10))

print("\nEjemplo de values_counts en frameworks:")
display(df_clean["frameworks"].value_counts(dropna=False).head(10))
```

---

### 2.5.3 Limpieza ligera de `company_size`

```python
df_clean["company_size"] = df_clean["company_size"].astype(str).str.strip()
df_clean["company_size"] = df_clean["company_size"].replace("nan", np.nan)

print("Valores únicos de company_size:")
print(sorted(df_clean["company_size"].dropna().unique()))
```

---

## 2.6 Conversión de tipos de datos

Las columnas `experience` y `salary_usd` deben ser numéricas para poder analizar nulos, outliers y estadísticas de manera correcta.

```python
df_clean["experience"] = pd.to_numeric(df_clean["experience"], errors="coerce")
df_clean["salary_usd"] = pd.to_numeric(df_clean["salary_usd"], errors="coerce")

print("Tipos de datos después de convertir experience y salary_usd:")
print(df_clean.dtypes)
```

---

## 2.7 Tratamiento de valores imposibles en variables numéricas

En esta etapa no se tratan aún los outliers por IQR, sino los **valores imposibles o claramente inválidos** según el dominio del problema.

### Reglas aplicadas

- `experience < 0`  → inválido
- `experience > 50` → inválido
- `salary_usd <= 0` → inválido

```python
# Conteos antes de corregir
exp_invalidas = ((df_clean["experience"] < 0) | (df_clean["experience"] > 50)).sum()
salary_invalidos = (df_clean["salary_usd"] <= 0).sum()

print("Cantidad de experiencias inválidas detectadas:", int(exp_invalidas))
print("Cantidad de salarios inválidos detectados:", int(salary_invalidos))

# Reemplazo por NaN
df_clean.loc[(df_clean["experience"] < 0) | (df_clean["experience"] > 50), "experience"] = np.nan
df_clean.loc[df_clean["salary_usd"] <= 0, "salary_usd"] = np.nan

print("\nNulos después de invalidar valores imposibles:")
print(df_clean[["experience", "salary_usd"]].isnull().sum())
```

---

## 2.8 Eliminación de duplicados

La eliminación de duplicados se hace **después** de normalizar texto y listas, porque así aparecen duplicados que antes parecían distintos por formato.

```python
duplicados_antes = df_clean.duplicated().sum()
print("Duplicados detectados antes de eliminar:", duplicados_antes)

df_clean = df_clean.drop_duplicates().reset_index(drop=True)

duplicados_despues = df_clean.duplicated().sum()
print("Duplicados después de eliminar:", duplicados_despues)
print("Nuevo shape:", df_clean.shape)
```

---

## 2.9 Manejo de valores nulos

En esta fase se aplica una estrategia distinta según el tipo e importancia de cada variable.

### Decisiones adoptadas

- `salary_usd`: eliminar filas con nulo porque es la variable principal del análisis
- `experience`: imputar con mediana
- `country`, `education`, `languages`, `frameworks`, `company_size`: completar con `Unknown`

```python
print("Nulos antes del tratamiento final:")
display(df_clean.isnull().sum())
```

---

### 2.9.1 Eliminar filas sin salario

```python
filas_antes = df_clean.shape[0]

df_clean = df_clean.dropna(subset=["salary_usd"]).copy()

filas_despues = df_clean.shape[0]
filas_eliminadas = filas_antes - filas_despues

print("Filas eliminadas por salary_usd nulo:", filas_eliminadas)
print("Shape después de eliminar salary_usd nulo:", df_clean.shape)
```

---

### 2.9.2 Imputar `experience` con mediana

```python
mediana_experience = df_clean["experience"].median()
print("Mediana de experience:", mediana_experience)

df_clean["experience"] = df_clean["experience"].fillna(mediana_experience)
df_clean["experience"] = df_clean["experience"].round().astype("Int64")

print("Nulos en experience después de imputar:", df_clean["experience"].isnull().sum())
print("Tipo final de experience:", df_clean["experience"].dtype)
```

---

### 2.9.3 Imputar categóricas con `Unknown`

```python
columnas_categoricas = ["country", "education", "languages", "frameworks", "company_size"]

for col in columnas_categoricas:
    df_clean[col] = df_clean[col].fillna("Unknown")

print("Nulos después de completar categóricas:")
display(df_clean[columnas_categoricas].isnull().sum())
```

---

## 2.10 Tratamiento de outliers

### 2.10.1 Revisar outliers de salario antes del capping

Se aplicará la técnica de **capping con IQR** para `salary_usd`, ya que permite conservar los registros pero limitando el efecto de valores extremos.

```python
resumen_salary_antes = resumen_outliers_iqr(df_clean, "salary_usd")

print("Resumen de outliers en salary_usd antes del capping:")
for k, v in resumen_salary_antes.items():
    print(f"{k}: {v}")
```

---

### 2.10.2 Aplicar capping a `salary_usd`

```python
# Guardar una copia para comparación visual
df_clean["salary_usd_before_cap"] = df_clean["salary_usd"].copy()

# Aplicar capping
df_clean["salary_usd"], lim_inf_salary, lim_sup_salary = cap_outliers_iqr(df_clean["salary_usd"])

print("Límite inferior aplicado:", lim_inf_salary)
print("Límite superior aplicado:", lim_sup_salary)
```

---

### 2.10.3 Verificar outliers de salario después del capping

```python
resumen_salary_despues = resumen_outliers_iqr(df_clean, "salary_usd")

print("Resumen de outliers en salary_usd después del capping:")
for k, v in resumen_salary_despues.items():
    print(f"{k}: {v}")
```

---

## 2.11 Validación antes vs después

Ahora se comparará el estado del dataset original contra el dataset limpio.

```python
resumen_comparativo = pd.DataFrame({
    "dataset": ["Original", "Limpio"],
    "filas": [df.shape[0], df_clean.shape[0]],
    "columnas": [df.shape[1], df_clean.shape[1]],
    "duplicados": [df.duplicated().sum(), df_clean.duplicated().sum()],
    "nulos_experience": [df["experience"].isnull().sum(), df_clean["experience"].isnull().sum()],
    "nulos_salary_usd": [df["salary_usd"].isnull().sum(), df_clean["salary_usd"].isnull().sum()]
})

display(resumen_comparativo)
```

> **Nota:** el dataset limpio tiene una columna adicional temporal llamada `salary_usd_before_cap`, creada solo para comparar el salario antes y después del tratamiento de outliers.

---

## 2.12 Revisión final de nulos y tipos

```python
print("Nulos finales por columna:")
display(df_clean.isnull().sum())

print("\nTipos de datos finales:")
print(df_clean.dtypes)

print("\nVista previa del dataset limpio:")
display(df_clean.head())
```

---

## 2.13 Visualización antes y después del tratamiento de outliers

### 2.13.1 Boxplot de salario antes y después

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.boxplot(x=df_clean["salary_usd_before_cap"], ax=axes[0], color="salmon")
axes[0].set_title("Salary USD antes del capping")

sns.boxplot(x=df_clean["salary_usd"], ax=axes[1], color="lightgreen")
axes[1].set_title("Salary USD después del capping")

plt.tight_layout()
plt.show()
```

---

### 2.13.2 Boxplot de experiencia final

```python
plt.figure(figsize=(8, 3))
sns.boxplot(x=df_clean["experience"], color="skyblue")
plt.title("Experience después de limpieza")
plt.show()
```

---

## 2.14 Resumen narrativo de la limpieza

### Puedes pegar este texto como explicación en markdown dentro del notebook

Durante la etapa de limpieza se identificaron y trataron múltiples problemas de calidad de datos. Primero se unificaron pseudonulos como `missing`, `N/A`, `null` y `?`, llevándolos a un formato estándar (`NaN`). Luego se normalizaron variables categóricas como país, nivel educativo, lenguajes y frameworks, corrigiendo diferencias de mayúsculas, espacios y repeticiones internas.

Posteriormente, se convirtieron a formato numérico las columnas `experience` y `salary_usd`, lo que permitió detectar valores imposibles como experiencia negativa, experiencia superior a 50 años o salarios menores o iguales a cero. Estos casos fueron tratados como datos inválidos.

En la limpieza de duplicados, primero se normalizaron los formatos para mejorar la detección y luego se eliminaron las filas repetidas. Para los valores faltantes, se decidió eliminar filas sin salario, ya que esta variable es central para el análisis, imputar la experiencia con la mediana y completar variables categóricas con la etiqueta `Unknown`.

Finalmente, los outliers de salario se trataron mediante **capping basado en IQR**, estrategia que conserva los registros pero reduce el impacto de valores extremos en el análisis.

---

## 2.15 Limpieza final para exportación

Como `salary_usd_before_cap` fue creada solo para comparar antes y después, al final conviene eliminarla antes de exportar.

```python
df_export = df_clean.drop(columns=["salary_usd_before_cap"]).copy()

print("Shape final a exportar:", df_export.shape)
display(df_export.head())
```

---

## 2.16 Exportación del dataset limpio

Si estás trabajando en Colab, puedes guardarlo así:

```python
df_export.to_csv("/content/software_developer_salary_clean.csv", index=False)
print("Archivo exportado correctamente en /content/software_developer_salary_clean.csv")
```

Si luego lo quieres bajar manualmente desde Colab:

```python
from google.colab import files
files.download("/content/software_developer_salary_clean.csv")
```

---

## 2.17 Conclusión de la etapa de limpieza

Con esta etapa, el dataset queda:

- sin duplicados exactos
- con tipos de datos adecuados
- con valores faltantes tratados
- con valores imposibles corregidos
- con outliers de salario controlados
- con variables categóricas más consistentes

Esto deja una base mucho más confiable para análisis, visualización, transformación y posible modelado posterior.