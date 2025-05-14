import numpy as np                        
import pandas as pd

import matplotlib.pyplot as plt              
from matplotlib import cm
import seaborn as sns

from io import BytesIO

import joblib
import streamlit as st

@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def preprocess_data(input_df):
    df = input_df.copy()
    
    df['tempo_emprego'] = df['tempo_emprego'].fillna(0)
    p99 = df['renda'].quantile(0.99)
    df['renda'] = np.where(df['renda'] > p99, p99, df['renda'])
    
    required_cols = ['idade', 'renda', 'tempo_emprego', 'posse_de_veiculo']
    return df[required_cols]

def main():
    st.set_page_config(
        page_title="Credit Scoring Tool",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("Sistema de Análise de Crédito")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📤 Carregamento de Dados")
        uploaded_file = st.file_uploader(
            "Arraste seu arquivo aqui", 
            type=['ftr', 'csv'],
        )
        
    if uploaded_file:
        try:
            df = pd.read_feather(uploaded_file) if uploaded_file.name.endswith('.ftr') else pd.read_csv(uploaded_file)
            df = df.sample(min(50000, len(df)))  
            
            @st.cache_resource
            def load_model():
                return joblib.load("C:/Users/amand/OneDrive/Desktop/Material Mod 38/modelo_final.pkl")
            
            model = load_model()
            
            processed_data = preprocess_data(df)
            
            scores = model.predict_proba(processed_data)[:, 1]
            df['score'] = np.round(scores, 4)
            df['risco'] = pd.cut(scores, bins=[0, 0.3, 0.7, 1], labels=['Baixo', 'Médio', 'Alto'])
            
            st.header("📊 Visão Geral")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Clientes", len(df))
            with col2:
                st.metric("Média Score", f"{df['score'].mean():.2%}")
            with col3:
                alto_risco = len(df[df['risco'] == 'Alto'])
                st.metric("Alto Risco", f"{alto_risco} ({alto_risco/len(df):.1%})")
            with col4:
                st.metric("Maior Score", f"{df['score'].max():.2%}")
            
            with st.expander("Filtros", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    risco_selecionado = st.multiselect(
                        "Nível de Risco",
                        options=['Baixo', 'Médio', 'Alto'],
                        default=['Alto', 'Médio']
                    )
                with col2:
                    score_min, score_max = st.slider(
                        "Faixa de Score",
                        min_value=0.0, max_value=1.0,
                        value=(0.5, 1.0),
                        format="%.2f"
                    )
            
            df_filtrado = df[
                (df['risco'].isin(risco_selecionado)) & 
                (df['score'].between(score_min, score_max))
            ]
            
            tab1, tab2 = st.tabs(["📈 Gráficos", "📋 Dados"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    risco_counts = df_filtrado['risco'].value_counts()
                    cores = {'Baixo': 'green', 'Médio': 'orange', 'Alto': 'red'}
                    ax.pie(
                        risco_counts, 
                        labels=risco_counts.index,
                        autopct='%1.1f%%',
                        colors=[cores[r] for r in risco_counts.index],
                        startangle=90
                    )
                    ax.set_title('Distribuição de Risco')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.histplot(
                        data=df_filtrado,
                        x='score',
                        hue='risco',
                        bins=20,
                        multiple='stack',
                        palette=cores
                    )
                    ax.set_title('Distribuição de Scores')
                    st.pyplot(fig)
                
                fig, ax = plt.subplots()
                sns.boxplot(
                    data=df_filtrado,
                    x='risco',
                    y='renda',
                    palette=cores,
                    order=['Baixo', 'Médio', 'Alto']
                )
                ax.set_title('Renda por Nível de Risco')
                st.pyplot(fig)
            
            with tab2:
                st.dataframe(df_filtrado.sort_values('score', ascending=False), height=500, hide_index=True)
                
                st.markdown("### 📤 Exportar Resultados")
                formato = st.radio("Formato:", ['CSV', 'Excel', 'Feather'], horizontal=True)
                
                if formato == 'CSV': st.download_button("Baixar CSV", df_filtrado.to_csv(index=False), "resultados.csv")
                elif formato == 'Excel':st.download_button("Baixar Excel", to_excel(df_filtrado), "resultados.xlsx")
                else: st.download_button("Baixar Feather", df_filtrado.to_feather(), "resultados.ftr")
        
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
            st.error("Verifique se o arquivo tem as colunas necessárias")

if __name__ == "__main__":
    main()