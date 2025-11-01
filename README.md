# 🩺 Predição de Câncer de Mama com Machine Learning

Este projeto utiliza **aprendizado de máquina supervisionado e não supervisionado** para **analisar dados clínicos e prever o diagnóstico de câncer de mama**.  
O sistema foi desenvolvido em **Python** com visualização interativa via **Streamlit**, e o modelo foi treinado utilizando dados do **Breast Cancer Dataset** (Scikit-learn).

---

## 🎯 Objetivo

Desenvolver um modelo capaz de prever se um tumor é **benigno ou maligno**, com base em medições extraídas de imagens digitais de biópsias de mama.  
Além disso, foram aplicadas técnicas de **clusterização (KMeans)** para identificar possíveis agrupamentos de pacientes com perfis semelhantes.

---

## ⚙️ Pipeline de Desenvolvimento

### 🔹 1. Coleta e Análise Exploratória (EDA)
- Leitura e inspeção dos dados (`load_breast_cancer` - scikit-learn)
- Estatísticas descritivas e análise de correlação
- Visualizações com Seaborn e Matplotlib:
  - Distribuição de classes
  - Heatmap de correlações
  - Violin plots e boxplots de outliers

### 🔹 2. Pré-processamento e Engenharia de Atributos
- Remoção de colunas irrelevantes
- Codificação da variável alvo (`LabelEncoder`)
- Normalização com `StandardScaler`
- Criação de novos atributos derivados:
  - Razões entre medidas (ex: `radius/area`)
  - Produtos entre variáveis de concavidade

### 🔹 3. Modelagem Supervisionada
Foram testados **9 algoritmos de Machine Learning**:
- Regressão Logística  
- Árvore de Decisão  
- Floresta Aleatória  
- Gradient Boosting  
- LightGBM  
- XGBoost  
- SVM  
- KNN  
- Naive Bayes  

> O modelo final foi obtido por meio de um **Ensemble (Voting Classifier)**, combinando os três melhores modelos.

### 🔹 4. Modelagem Não Supervisionada
- Aplicação de **PCA (Análise de Componentes Principais)**  
- Clusterização com **KMeans (k=2)**  
- Visualização dos agrupamentos e comparação com os rótulos reais

### 🔹 5. Interpretação e Explicabilidade
- Uso de **SHAP (SHapley Additive Explanations)** para compreender as variáveis mais influentes
- Geração de gráficos:
  - SHAP Feature Importance
  - SHAP Summary Plot

### 🔹 6. Avaliação Final
- Métricas principais:
  - **Accuracy:** 98.25%
  - **Precision:** 100%
  - **Recall:** 95.24%
  - **F1-Score:** 97.56%
  - **ROC-AUC:** 99.70%
- Curvas ROC e Precision-Recall
- Análise de erros e métricas clínicas (Sensibilidade, Especificidade, PPV, NPV)

---

## 🧠 Tecnologias Utilizadas

| Categoria | Ferramentas |
|:-----------|:-------------|
| Linguagem | Python 3.10 |
| ML Frameworks | Scikit-learn, XGBoost, LightGBM |
| Visualização | Matplotlib, Seaborn |
| Explicabilidade | SHAP |
| Interface | Streamlit |
| Ambiente | Google Colab + Streamlit Cloud |

---

## 📁 Estrutura do Projeto

📦 breast_cancer_prediction/
┣ 📜 app.py → Aplicação Streamlit
┣ 📜 main_notebook.ipynb → Treinamento e análise completa
┣ 📁 artifacts/
┃ ┣ 📜 best_model.pkl → Modelo final salvo
┃ ┗ 📜 scaler.pkl → Escalador padrão (StandardScaler)
┣ 📜 requirements.txt → Dependências do projeto
┗ 📜 README.md → Documentação do projeto


---

## 🚀 Execução Local

### 🔧 Instalar dependências
```bash
pip install -r requirements.txt

▶️ Rodar o app Streamlit

streamlit run app.py

🌐 Deploy Online

Aplicação hospedada no Streamlit Cloud:
👉 https://ml-breast-cancer-yourname.streamlit.app

📜 Licença

Este projeto é distribuído sob a licença MIT.
Sinta-se à vontade para usar e modificar para fins educacionais ou de pesquisa.

👩‍💻 Autoria

Maria Vitória
Estudante de Análise e Desenvolvimento de Sistemas — Faculdade Senac Pernambuco
Contato: vitoriagomes1510@gmail.com

🌎 Recife - PE
