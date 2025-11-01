import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# ==============================================
# 🧠 APP: Predição de Câncer de Mama (Versão 2)
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
FEATURE_IMG_PATH = "feature_importance.png"

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
    "mean radius": st.sidebar.number_input("Raio médio", 0.0, 30.0, 14.0),
    "mean texture": st.sidebar.number_input("Textura média", 0.0, 40.0, 20.0),
    "mean perimeter": st.sidebar.number_input("Perímetro médio", 0.0, 200.0, 90.0),
    "mean area": st.sidebar.number_input("Área média", 0.0, 2500.0, 700.0),
    "mean smoothness": st.sidebar.number_input("Suavidade média", 0.0, 1.0, 0.1),
    "mean compactness": st.sidebar.number_input("Compacidade média", 0.0, 1.0, 0.2),
    "mean concavity": st.sidebar.number_input("Concavidade média", 0.0, 1.0, 0.3),
    "mean concave points": st.sidebar.number_input("Pontos côncavos médios", 0.0, 1.0, 0.15),
    "mean symmetry": st.sidebar.number_input("Simetria média", 0.0, 1.0, 0.2),
    "mean fractal dimension": st.sidebar.number_input("Dimensão fractal média", 0.0, 1.0, 0.06),
}

# ==============================================
# ⚙️ Pré-processamento ajustado
# ==============================================
try:
    all_features = list(scaler.feature_names_in_)
    full_input = pd.DataFrame(columns=all_features)
    full_input.loc[0] = np.zeros(len(all_features))

    for col, val in inputs.items():
        if col in full_input.columns:
            full_input.at[0, col] = val

    scaled_input = scaler.transform(full_input)
except Exception as e:
    st.error("❌ Erro ao preparar os dados para previsão.")
    st.code(str(e))
    st.stop()

# ==============================================
# 🔮 Predição + Visualização
# ==============================================
if st.button("🔍 Realizar Previsão"):

    if np.isnan(scaled_input).any():
        st.error("❌ Existem valores inválidos. Verifique as entradas.")
        st.stop()

    try:
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]

        benigno_prob = proba[1] * 100
        maligno_prob = proba[0] * 100

        st.markdown("---")

        if prediction == 1:
            st.success(f"🟢 **Benigno** ({benigno_prob:.2f}% de confiança)")
            st.progress(int(benigno_prob))
        else:
            st.error(f"🔴 **Maligno** ({maligno_prob:.2f}% de confiança)")
            st.progress(int(maligno_prob))

        # 📊 Exibir gráfico de barras com probabilidades
        st.markdown("### 📊 Distribuição das probabilidades:")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Benigno", "Maligno"], [benigno_prob, maligno_prob], color=["green", "red"], alpha=0.7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Confiança (%)")
        ax.set_title("Predição do Modelo")
        st.pyplot(fig)

        # 📈 Mostrar gráfico de importância de features (se existir)
        if os.path.exists(FEATURE_IMG_PATH):
            st.markdown("### 🔍 Importância das Principais Features")
            st.image(FEATURE_IMG_PATH, caption="Importância das variáveis no modelo", use_container_width=True)

        st.markdown("---")

        # ==============================================
        # 📄 Gerar Relatório PDF
        # ==============================================
        st.subheader("📄 Gerar Relatório da Previsão")

        if st.button("🧾 Baixar Relatório PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Relatório de Predição - Câncer de Mama", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Resultado: {'Benigno' if prediction == 1 else 'Maligno'}", ln=True)
            pdf.cell(0, 10, f"Confiança: {max(benigno_prob, maligno_prob):.2f}%", ln=True)
            pdf.cell(0, 10, f"Modelo: Ensemble (Voting Classifier)", ln=True)
            pdf.ln(10)
            pdf.multi_cell(0, 8, "Variáveis informadas:\n" + "\n".join([f"{k}: {v}" for k, v in inputs.items()]))

            # Salvar o PDF em memória
            buffer = BytesIO()
            pdf.output(buffer)
            buffer.seek(0)

            st.download_button(
                label="📥 Baixar PDF",
                data=buffer,
                file_name="relatorio_predicao.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error("⚠️ Erro ao executar a previsão.")
        st.code(str(e))

# ==============================================
# ℹ️ Informações do modelo
# ==============================================
with st.expander("ℹ️ Sobre o Modelo"):
    st.write(
        """
        - **Tipo:** Ensemble (Voting Classifier)  
        - **Algoritmos:** SVM, Regressão Logística, LightGBM e Random Forest  
        - **Acurácia:** 97.37%  
        - **F1-Score:** 97.37%  
        - **ROC-AUC:** 99.47%  
        """
    )

st.markdown("---")
st.caption("Desenvolvido por **Maria Vitória Gomes** • Faculdade Senac Pernambuco 💙")
