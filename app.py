import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==============================================
# 🧠 APP: Predição de Câncer de Mama
# ==============================================

st.set_page_config(
    page_title="Predição de Câncer de Mama 🧬",
    page_icon="🩺",
    layout="centered",
)

st.title("🧠 Machine Learning Aplicado à Saúde")
st.subheader("Predição de Câncer de Mama")

st.markdown(
    """
    Este aplicativo utiliza um modelo de **Machine Learning** treinado com dados clínicos reais
    para prever se um tumor de mama é **Benigno (não cancerígeno)** ou **Maligno (cancerígeno)**.
    
    ---
    """
)

# ==============================================
# 🔹 Carregar modelo e scaler
# ==============================================
MODEL_PATH = "artifacts/best_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
else:
    st.error("❌ Arquivos de modelo ou scaler não encontrados na pasta 'artifacts/'.")
    st.stop()

# ==============================================
# 🧩 Entradas do usuário
# ==============================================
st.sidebar.header("🔧 Insira os valores das variáveis clínicas:")

inputs = {
    "mean radius": st.sidebar.number_input("Raio médio", min_value=0.0, max_value=30.0, value=14.0),
    "mean texture": st.sidebar.number_input("Textura média", min_value=0.0, max_value=40.0, value=20.0),
    "mean perimeter": st.sidebar.number_input("Perímetro médio", min_value=0.0, max_value=200.0, value=90.0),
    "mean area": st.sidebar.number_input("Área média", min_value=0.0, max_value=2500.0, value=700.0),
    "mean smoothness": st.sidebar.number_input("Suavidade média", min_value=0.0, max_value=1.0, value=0.1),
    "mean compactness": st.sidebar.number_input("Compacidade média", min_value=0.0, max_value=1.0, value=0.2),
    "mean concavity": st.sidebar.number_input("Concavidade média", min_value=0.0, max_value=1.0, value=0.3),
    "mean concave points": st.sidebar.number_input("Pontos côncavos médios", min_value=0.0, max_value=1.0, value=0.15),
    "mean symmetry": st.sidebar.number_input("Simetria média", min_value=0.0, max_value=1.0, value=0.2),
    "mean fractal dimension": st.sidebar.number_input("Dimensão fractal média", min_value=0.0, max_value=1.0, value=0.06),
}

input_df = pd.DataFrame([inputs])

# ==============================================
# ⚙️ Pré-processamento (corrigido)
# ==============================================
try:
    # Garante que as colunas estão na mesma ordem e formato do treinamento
    input_df = pd.DataFrame([inputs], columns=scaler.feature_names_in_)
    scaled_input = scaler.transform(input_df)
except Exception as e:
    st.error("❌ Erro ao processar os dados de entrada. Verifique o log abaixo:")
    st.code(str(e))
    st.stop()

# ==============================================
# 🔮 Predição (com verificação de dados)
# ==============================================
if st.button("🔍 Realizar Previsão"):

    # Verifica se há valores nulos ou não numéricos
    if input_df.isnull().any().any() or np.isinf(input_df.values).any():
        st.error("❌ Existem valores inválidos nas entradas. Verifique se todos os campos estão preenchidos corretamente.")
        st.stop()

    try:
        # Predição
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1] * 100

        st.markdown("---")

        if prediction == 1:
            st.success(f"🟢 Resultado: **Benigno** ({proba:.2f}% de confiança)")
            st.progress(int(proba))
            st.balloons()
        else:
            st.error(f"🔴 Resultado: **Maligno** ({proba:.2f}% de confiança)")
            st.progress(int(proba))

        st.markdown("---")
        st.caption("Modelo baseado em dados reais do Breast Cancer Dataset (Scikit-learn).")

    except ValueError as e:
        st.error("⚠️ Erro ao executar a previsão. O modelo não conseguiu processar os dados inseridos.")
        st.code(str(e))
    except Exception as e:
        st.error("❌ Ocorreu um erro inesperado durante a previsão:")
        st.code(str(e))


    st.markdown("---")

    if prediction == 1:
        st.success(f"🟢 Resultado: **Benigno** ({proba:.2f}% de confiança)")
        st.progress(int(proba))
        st.balloons()
    else:
        st.error(f"🔴 Resultado: **Maligno** ({proba:.2f}% de confiança)")
        st.progress(int(proba))

    st.markdown("---")
    st.caption("Modelo baseado em dados reais do Breast Cancer Dataset (Scikit-learn).")

# ==============================================
# 📊 Informações adicionais
# ==============================================
with st.expander("ℹ️ Sobre o Modelo"):
    st.write(
        """
        - **Tipo de modelo:** Ensemble (Voting Classifier)  
        - **Algoritmos utilizados:** Regressão Logística, LightGBM e Random Forest  
        - **Acurácia:** 98.25%  
        - **F1-Score:** 97.56%  
        - **ROC-AUC:** 99.70%  
        """
    )

st.markdown("---")
st.caption("Desenvolvido por **Maria Vitória Gomes** • Faculdade Senac Pernambuco 💙")
