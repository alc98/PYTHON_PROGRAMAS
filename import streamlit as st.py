import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(
    page_title="Demo Análisis de Sentimiento",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Demo de Análisis de Sentimiento en Reviews")
st.markdown("""
Esta app usa el **modelo entrenado en `Reviews_sentiment_advanced.ipynb`** para clasificar
reviews como **POSITIVAS** o **NO POSITIVAS**.

- Modo 1: analizar una review individual.
- Modo 2: subir un CSV con muchas reviews y clasificarlas todas.
""")


# =========================
# CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        st.error(f"No se encontró el fichero de modelo: `{model_path}`")
        st.stop()
    return joblib.load(path)

# 🔴 AJUSTA AQUÍ el nombre de tu fichero .pkl
MODEL_PATH = "best_sentiment_pipeline.pkl"
model = load_model(MODEL_PATH)


# =====================================================
# FUNCIONES AUXILIARES PARA FEATURES Y PREDICCIONES
# =====================================================
def build_features_from_text(
    text: str,
    help_num: float = 0.0,
    help_den: float = 0.0
) -> pd.DataFrame:
    """
    Construye un DataFrame con las mismas columnas que espera el pipeline
    a partir de la review en texto y, opcionalmente, info de helpfulness.
    Ajusta aquí si tu pipeline espera más/menos columnas.
    """
    text = text or ""
    review_len_chars = len(text)
    review_len_words = len(text.split()) if text.strip() else 0

    help_num = help_num or 0.0
    help_den = help_den or 0.0
    help_ratio = help_num / (help_den + 1.0)

    # 🔴 AJUSTA ESTAS COLUMNAS A LO QUE ESPERE TU PIPELINE
    data = {
        "full_text": text,
        "review_len_chars": review_len_chars,
        "review_len_words": review_len_words,
        "help_num": help_num,
        "help_den": help_den,
        "help_ratio": help_ratio,
    }

    return pd.DataFrame([data])


def predict_single(df_row: pd.DataFrame):
    """
    Lanza predicción para un único registro.
    Devuelve etiqueta (0/1), texto legible y probabilidad si está disponible.
    """
    y_pred = model.predict(df_row)[0]

    proba_pos = None
    if hasattr(model, "predict_proba"):
        proba_pos = float(model.predict_proba(df_row)[0, 1])
    elif hasattr(model, "decision_function"):
        # Decision function → lo pasamos por una sigmoide aproximada
        score = model.decision_function(df_row)[0]
        proba_pos = 1 / (1 + np.exp(-score))

    label = "POSITIVO" if y_pred == 1 else "NO POSITIVO"
    return y_pred, label, proba_pos


def map_label(y_pred):
    return np.where(y_pred == 1, "POSITIVO", "NO POSITIVO")


# =========================
# SIDEBAR: MODO DE USO
# =========================
st.sidebar.header("⚙️ Opciones de la demo")
mode = st.sidebar.radio(
    "Selecciona modo de uso:",
    ["Review individual", "CSV con muchas reviews"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Modelo cargado desde: `{MODEL_PATH}`")


# =========================
# MODO 1: REVIEW INDIVIDUAL
# =========================
if mode == "Review individual":
    st.subheader("📝 Analizar una review individual")

    col1, col2 = st.columns([2, 1])

    with col1:
        review_text = st.text_area(
            "Escribe o pega aquí la review del cliente:",
            height=200,
            placeholder="Ejemplo: The product arrived on time and works perfectly. Very satisfied."
        )

    with col2:
        st.markdown("### Parámetros opcionales de utilidad")

        help_num = st.number_input(
            "Helpfulness numerator (votos útiles)",
            min_value=0.0, value=0.0, step=1.0
        )
        help_den = st.number_input(
            "Helpfulness denominator (total votos)",
            min_value=0.0, value=0.0, step=1.0
        )

        st.markdown("Estos campos imitan las columnas de **Helpfulness** "
                    "del dataset original. Si no los conoces, déjalos en 0.")

    if st.button("🔍 Analizar sentimiento", type="primary"):
        if not review_text.strip():
            st.warning("Por favor, introduce una review antes de analizar.")
        else:
            # Construir features
            X_new = build_features_from_text(
                text=review_text,
                help_num=help_num,
                help_den=help_den
            )

            # Mostrar features calculadas (visión técnica ligera)
            with st.expander("Ver features calculadas (visión técnica)"):
                st.write(X_new)

            # Predicción
            y_pred, label, proba_pos = predict_single(X_new)

            st.markdown("---")
            st.subheader("📌 Resultado")

            if label == "POSITIVO":
                st.success(f"Sentimiento detectado: **{label}**")
            else:
                st.error(f"Sentimiento detectado: **{label}**")

            if proba_pos is not None:
                st.write(f"Probabilidad de review **positiva**: **{proba_pos:.2%}**")

            # Pequeño resumen explicativo para negocio
            st.markdown("""
            **Interpretación para negocio:**
            - Si el modelo la clasifica como POSITIVO con alta probabilidad, es una opinión alineada con clientes satisfechos.
            - Si aparece como NO POSITIVO, conviene revisar el texto para detectar quejas, problemas o fricciones.
            """)


# =========================
# MODO 2: CSV CON MUCHAS REVIEWS
# =========================
else:
    st.subheader("📁 Analizar muchas reviews desde un CSV")

    st.markdown("""
    El CSV debe contener al menos una columna de texto que podamos usar como review.
    Idealmente, será el mismo formato de **Reviews.csv** que usaste en el notebook
    (`Summary`, `Text`, etc.).

    Puedes:
    1. Subir el CSV original y dejar que la app construya `full_text` y features numéricas.
    2. O subir un CSV ya preprocesado con las columnas que espera el pipeline.
    """)

    uploaded_file = st.file_uploader("Sube un fichero CSV", type=["csv"])

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        text_col = st.text_input(
            "Nombre de la columna de texto principal (por ejemplo, 'Text' o 'full_text')",
            value="Text"
        )
    with col_cfg2:
        summary_col = st.text_input(
            "Columna opcional de summary/título (por ejemplo, 'Summary')",
            value="Summary"
        )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Primeras filas del CSV subido:")
        st.dataframe(df.head())

        # Botón para lanzar predicciones
        if st.button("🚀 Clasificar todas las reviews", type="primary"):
            if text_col not in df.columns and "full_text" not in df.columns:
                st.error(
                    f"No se encontró la columna '{text_col}' ni 'full_text' en el CSV. "
                    "Ajusta el nombre de la columna de texto."
                )
            else:
                # Si no hay full_text, la creamos a partir de summary + text (si existen)
                if "full_text" not in df.columns:
                    base_text = df[text_col].fillna("").astype(str)
                    if summary_col in df.columns:
                        base_sum = df[summary_col].fillna("").astype(str)
                        df["full_text"] = base_sum + " " + base_text
                    else:
                        df["full_text"] = base_text

                # Creamos las features numéricas si no están
                if "review_len_chars" not in df.columns:
                    df["review_len_chars"] = df["full_text"].fillna("").astype(str).str.len()
                if "review_len_words" not in df.columns:
                    df["review_len_words"] = df["full_text"].fillna("").astype(str).apply(
                        lambda x: len(x.split())
                    )

                if "help_num" not in df.columns:
                    df["help_num"] = 0.0
                if "help_den" not in df.columns:
                    df["help_den"] = 0.0
                if "help_ratio" not in df.columns:
                    df["help_ratio"] = df["help_num"] / (df["help_den"] + 1.0)

                # Seleccionamos sólo las columnas que espera el modelo
                # 🔴 AJUSTA ESTO A TU PIPELINE:
                cols_for_model = [
                    "full_text",
                    "review_len_chars",
                    "review_len_words",
                    "help_num",
                    "help_den",
                    "help_ratio",
                ]
                X = df[cols_for_model]

                # Predicción en bloque
                y_pred = model.predict(X)
                labels = map_label(y_pred)

                df["sentiment_pred"] = y_pred
                df["sentiment_label"] = labels

                proba_pos = None
                if hasattr(model, "predict_proba"):
                    proba_pos = model.predict_proba(X)[:, 1]
                    df["sentiment_proba_pos"] = proba_pos

                st.markdown("### ✅ Resultados")
                st.write(df.head())

                # Resumen agregados para negocio
                st.markdown("### 📊 Resumen de sentimiento")
                st.write(df["sentiment_label"].value_counts())

                st.bar_chart(df["sentiment_label"].value_counts())

                if proba_pos is not None:
                    st.markdown("### Distribución de probabilidad de clase positiva")
                    st.histogram = st.pyplot  # Para no romper si no hay figura previa
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()
                    ax.hist(proba_pos, bins=20)
                    ax.set_xlabel("Probabilidad de review positiva")
                    ax.set_ylabel("Número de reviews")
                    ax.set_title("Distribución de probabilidad (clase positiva)")
                    st.pyplot(fig)

                # Opción de descarga
                st.markdown("### 💾 Descargar resultados enriquecidos")
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Descargar CSV con predicciones",
                    data=csv_out,
                    file_name="reviews_con_sentiment_predicho.csv",
                    mime="text/csv"
                )

                st.markdown("""
                **Interpretación para negocio:**
                - Puedes ver cuántas reviews son positivas vs no positivas de un vistazo.
                - El fichero descargable permite seguir trabajando en Excel/Power BI/Tableau.
                - Así puedes cruzar el sentimiento con producto, país, canal, etc.
                """)

