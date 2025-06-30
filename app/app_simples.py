"""
Dashboard Streamlit SIMPLIFICADO - Versão para Apresentação
Usa apenas dados em cache para evitar travamentos.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Análise de Avaliações de Livros",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .success-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_cache_data():
    """Carrega dados dos arquivos de cache."""
    cache_data = {}
    cache_path = Path("data/cache")
    
    if cache_path.exists():
        # Tentar carregar diferentes arquivos de cache
        cache_files = [
            "llm_analysis_results.json",
            "business_recommendations.json", 
            "book_performance_50.json",
            "user_insights_10000.json",
            "sentiment_trends_2000.json"
        ]
        
        for file in cache_files:
            file_path = cache_path / file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data[file.replace('.json', '')] = json.load(f)
                except Exception as e:
                    st.warning(f"Erro ao carregar {file}: {e}")
    
    return cache_data

def create_metric_card(label, value):
    """Cria card de métrica."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div>{label}</div>
    </div>
    """, unsafe_allow_html=True)

def show_llm_results(cache_data):
    """Mostra resultados da análise LLM."""
    if 'llm_analysis_results' in cache_data:
        results = cache_data['llm_analysis_results']
        
        st.markdown("### 🧠 Análise LLM Híbrida - Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### VADER (Lexical)")
            if 'local_analysis' in results:
                local = results['local_analysis']
                st.metric("Textos Processados", f"{local.get('total_processed', 0):,}")
                st.metric("Tempo", local.get('processing_time', 'N/A'))
                
        with col2:
            st.markdown("#### DistilBERT (Neural)")
            if 'comparison' in results:
                comp = results['comparison']
                st.metric("Concordância VADER", f"{comp.get('agreement_rate', 0)*100:.1f}%")
                st.metric("Discrepância Média", f"{comp.get('vader_vs_local', {}).get('local_avg_discrepancy', 0):.3f}")
                
        with col3:
            st.markdown("#### GPT (Qualitativo)")
            if 'gpt_analysis' in results:
                gpt = results['gpt_analysis']
                st.metric("Textos Analisados", gpt.get('total_processed', 0))
                st.metric("Confiança Média", f"{gpt.get('avg_confidence', 0)*100:.1f}%")
        
        # Insights principais
        if 'insights' in results:
            st.markdown("#### 📊 Insights Principais")
            for insight in results['insights']:
                st.markdown(f"• {insight}")
        
        # Temas extraídos pelo GPT
        if 'gpt_analysis' in results and 'themes_extracted' in results['gpt_analysis']:
            st.markdown("#### 🎯 Temas Identificados pelo GPT")
            themes = results['gpt_analysis']['themes_extracted']
            
            # Criar gráfico de temas
            fig = px.bar(
                x=themes[:10],  # Top 10 temas
                y=[1]*len(themes[:10]),  # Frequência igual para visualização
                title="Top 10 Temas Extraídos",
                labels={'x': 'Temas', 'y': 'Frequência'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_business_recommendations(cache_data):
    """Mostra recomendações de negócio."""
    if 'business_recommendations' in cache_data:
        rec = cache_data['business_recommendations']
        
        st.markdown("### 💼 Recomendações Estratégicas")
        
        # Quick Wins
        if 'quick_wins' in rec:
            st.markdown("#### 🚀 Quick Wins")
            for i, win in enumerate(rec['quick_wins'][:4], 1):
                st.markdown(f"**{i}.** {win}")
        
        # Impacto Estimado
        if 'estimated_impact' in rec:
            st.markdown("#### 📈 Impacto Estimado")
            impact = rec['estimated_impact']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Aumento de Receita", impact.get('revenue_increase', 'N/A'))
                st.metric("Retenção de Usuários", impact.get('user_retention', 'N/A'))
            with col2:
                st.metric("Eficiência Operacional", impact.get('operational_efficiency', 'N/A'))  
                st.metric("Redução Tempo Insights", impact.get('time_to_insights', 'N/A'))

def show_performance_data(cache_data):
    """Mostra dados de performance dos livros."""
    if 'book_performance_50' in cache_data:
        books = cache_data['book_performance_50']
        
        if 'top_books' in books:
            st.markdown("### 📚 Top Livros por Performance")
            
            top_books = books['top_books'][:10]  # Top 10
            
            # Criar DataFrame para visualização
            df_books = pd.DataFrame(top_books)
            
            if not df_books.empty:
                # Gráfico de barras horizontais
                fig = px.bar(
                    df_books, 
                    x='avg_score', 
                    y='title',
                    orientation='h',
                    title="Top 10 Livros - Score Médio",
                    labels={'avg_score': 'Score Médio', 'title': 'Livro'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela detalhada
                st.markdown("#### 📋 Detalhes dos Top Livros")
                display_df = df_books[['title', 'author', 'avg_score', 'review_count']].head(10)
                display_df.columns = ['Título', 'Autor', 'Score Médio', 'Nº Reviews']
                st.dataframe(display_df, use_container_width=True)

def main():
    """Função principal do dashboard."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Análise Inteligente de Avaliações de Livros</h1>', unsafe_allow_html=True)
    
    # Carregar dados do cache
    cache_data = load_cache_data()
    
    if not cache_data:
        st.error("❌ Nenhum dado em cache encontrado. Execute primeiro o processamento.")
        st.info("💡 Execute: `python run.py` para gerar os dados.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Simplificado")
    st.sidebar.success("✅ Dados carregados do cache")
    st.sidebar.info(f"📁 Arquivos encontrados: {len(cache_data)}")
    
    # Status dos dados
    st.markdown("### 📊 Status do Sistema")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_metric_card("💾 Arquivos Cache", len(cache_data))
    with col2:
        create_metric_card("🧠 Análise LLM", "✅ Disponível" if 'llm_analysis_results' in cache_data else "❌ Indisponível")
    with col3:
        create_metric_card("💼 Recomendações", "✅ Disponível" if 'business_recommendations' in cache_data else "❌ Indisponível")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧠 Análise LLM",
        "💼 Recomendações", 
        "📚 Performance Livros"
    ])
    
    with tab1:
        show_llm_results(cache_data)
    
    with tab2:
        show_business_recommendations(cache_data)
    
    with tab3:
        show_performance_data(cache_data)
    
    # Rodapé
    st.markdown("---")
    st.markdown("### 🎯 Sistema Pronto para Apresentação")
    st.markdown("""
    <div class="success-box">
        <strong>✅ Dashboard funcionando com dados em cache</strong><br>
        • Análise LLM híbrida processada<br>
        • Recomendações estratégicas disponíveis<br>
        • Performance dos livros analisada<br>
        • Pronto para demonstração técnica
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()