#!/usr/bin/env python3
"""
Processador Otimizado de Dados para Análise de Avaliações de Livros
Versão otimizada para processamento de grandes volumes de dados
"""

import pandas as pd
import numpy as np
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Silenciar warnings do pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedDataProcessor:
    """
    Processador otimizado para grandes volumes de dados de avaliações de livros.
    Utiliza processamento em chunks para evitar problemas de memória.
    """
    
    def __init__(self, input_file: str = "Books_rating.csv", output_dir: str = "data"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurações de processamento
        self.chunk_size = 50000  # Processar 50k linhas por vez
        self.max_samples = 25000  # Máximo de amostras para análise
        self.sample_ratio = 0.1  # 10% de amostra por chunk
        
    def analyze_file_structure(self) -> Dict[str, Any]:
        """
        Analisa a estrutura do arquivo para otimizar o processamento.
        
        Returns:
            Informações sobre a estrutura do arquivo
        """
        logger.info("Fase 1: Análise inicial dos dados")
        logger.info("Analisando estrutura do arquivo...")
        
        # Ler apenas as primeiras linhas para análise
        sample_df = pd.read_csv(self.input_file, nrows=1000)
        
        analysis = {
            'total_columns': len(sample_df.columns),
            'columns': list(sample_df.columns),
            'dtypes': sample_df.dtypes.to_dict(),
            'memory_usage': sample_df.memory_usage(deep=True).sum(),
            'sample_size': len(sample_df)
        }
        
        logger.info(f"Análise concluída: {len(sample_df)} linhas analisadas")
        return analysis
    
    def process_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> pd.DataFrame:
        """
        Processa um chunk de dados com limpeza e otimização.
        
        Args:
            chunk: DataFrame com dados do chunk
            chunk_num: Número do chunk para logging
            
        Returns:
            DataFrame processado
        """
        logger.info(f"Processando chunk {chunk_num}")
        
        # Limpeza básica
        chunk = chunk.copy()
        
        # Converter colunas de texto para string
        text_columns = ['summary', 'text']
        for col in text_columns:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).str.strip()
        
        # Remover linhas com dados inválidos
        chunk = chunk.dropna(subset=['score', 'User_id'])
        
        # Filtrar scores válidos (1-5)
        chunk = chunk[chunk['score'].between(1, 5)]
        
        # Converter timestamp se existir
        if 'time' in chunk.columns:
            chunk['time'] = pd.to_datetime(chunk['time'], unit='s', errors='coerce')
            chunk['year_review'] = chunk['time'].dt.year
            chunk['month_review'] = chunk['time'].dt.month
        
        # Amostragem inteligente por score para manter distribuição
        if len(chunk) > self.max_samples * self.sample_ratio:
            sampled = chunk.groupby('score', group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(self.max_samples * self.sample_ratio / 5)))
            )
            chunk = sampled.reset_index(drop=True)
        
        return chunk
    
    def process_data(self) -> pd.DataFrame:
        """
        Processa o arquivo completo em chunks.
        
        Returns:
            DataFrame consolidado com dados processados
        """
        logger.info("Fase 2: Processamento por chunks")
        chunks = []
        chunk_num = 1
        total_rows = 0

        for chunk in pd.read_csv(self.input_file, chunksize=self.chunk_size):
            processed_chunk = self.process_chunk(chunk, chunk_num)
            if len(processed_chunk) > 0:
                chunks.append(processed_chunk)
                total_rows += len(processed_chunk)
            print(f"[INFO] Chunk {chunk_num} processado ({len(processed_chunk)} linhas, total acumulado: {total_rows})")
            chunk_num += 1

        logger.info("Fase 3: Consolidação e otimização")
        logger.info("Consolidando chunks...")
        
        # Consolidar todos os chunks
        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            
            # Amostragem final se necessário
            if len(final_df) > self.max_samples:
                final_df = final_df.sample(n=self.max_samples, random_state=42)
            
            # Adicionar métricas de texto
            final_df['text_length'] = final_df['text'].str.len()
            final_df['has_text'] = final_df['text'].str.len() > 10
            
            # NLP: Análise de sentimento em amostra estratificada de 10.000 textos
            logger.info("Executando análise de sentimento NLP (VADER) em amostra de 10.000 textos...")
            analyzer = SentimentIntensityAnalyzer()
            def get_sentiment(text):
                if not isinstance(text, str) or not text.strip():
                    return 'neutral'
                score = analyzer.polarity_scores(text)['compound']
                if score >= 0.05:
                    return 'positive'
                elif score <= -0.05:
                    return 'negative'
                else:
                    return 'neutral'
            # Amostragem estratificada por score
            sample_nlp = final_df.groupby('score', group_keys=False).apply(lambda x: x.sample(min(len(x), 2000), random_state=42))
            sample_nlp = sample_nlp.reset_index(drop=True)
            sample_nlp['sentiment'] = sample_nlp['text'].apply(get_sentiment)
            # Merge resultado para o DataFrame final (apenas para as linhas amostradas)
            final_df = final_df.merge(sample_nlp[['Id', 'User_id', 'sentiment']], on=['Id', 'User_id'], how='left')
            logger.info("Coluna 'sentiment' adicionada para amostra de 10.000 textos.")
            logger.info(f"Distribuição de sentimento na amostra: {final_df['sentiment'].value_counts(dropna=False).to_dict()}")
            return final_df
        else:
            logger.error("Nenhum chunk válido encontrado")
            return pd.DataFrame()
    
    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Salva os dados processados e gera análises.
        
        Args:
            df: DataFrame processado
        """
        logger.info("Fase 4: Salvando dados processados")
        
        # Salvar dados processados
        output_file = self.output_dir / "merged_data_clean.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Dados salvos em: {output_file}")
        
        # Gerar análise dos dados
        analysis = self.generate_data_analysis(df)
        analysis_file = self.output_dir / "data_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Análise salva em: {analysis_file}")
        
        # Gerar estatísticas
        stats = self.generate_statistics(df)
        stats_file = self.output_dir / "data_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Estatísticas salvas em: {stats_file}")
        
        # Gerar resultados de análise
        results = self.generate_analysis_results(df)
        results_file = self.output_dir / "analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Resultados salvos em: {results_file}")
    
    def generate_data_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera análise detalhada dos dados processados.
        
        Args:
            df: DataFrame processado
            
        Returns:
            Análise dos dados
        """
        analysis = {
            'data_overview': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'processing_date': datetime.now().isoformat()
            },
            'score_distribution': df['score'].value_counts().sort_index().to_dict(),
            'text_analysis': {
                'avg_text_length': float(df['text_length'].mean()),
                'max_text_length': int(df['text_length'].max()),
                'min_text_length': int(df['text_length'].min()),
                'texts_with_content': int(df['has_text'].sum()),
                'texts_without_content': int((~df['has_text']).sum())
            },
            'user_analysis': {
                'unique_users': int(df['User_id'].nunique()),
                'avg_reviews_per_user': float(len(df) / df['User_id'].nunique())
            }
        }
        
        # Análise temporal se disponível
        if 'year_review' in df.columns:
            analysis['temporal_analysis'] = {
                'year_distribution': df['year_review'].value_counts().sort_index().to_dict(),
                'month_distribution': df['month_review'].value_counts().sort_index().to_dict()
            }
        
        return analysis
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera estatísticas descritivas dos dados.
        
        Args:
            df: DataFrame processado
            
        Returns:
            Estatísticas descritivas
        """
        stats = {
            'score_stats': {
                'mean': float(df['score'].mean()),
                'median': float(df['score'].median()),
                'std': float(df['score'].std()),
                'min': float(df['score'].min()),
                'max': float(df['score'].max())
            },
            'text_length_stats': {
                'mean': float(df['text_length'].mean()),
                'median': float(df['text_length'].median()),
                'std': float(df['text_length'].std()),
                'min': int(df['text_length'].min()),
                'max': int(df['text_length'].max())
            }
        }
        
        return stats
    
    def generate_analysis_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera resultados de análise para o dashboard.
        
        Args:
            df: DataFrame processado
            
        Returns:
            Resultados de análise
        """
        logger.info("Gerando resultados de análise...")
        
        # Top livros por score
        top_books = df.groupby('Id').agg({
            'score': ['mean', 'count'],
            'Title': 'first'
        }).round(3)
        top_books.columns = ['avg_score', 'review_count', 'title']
        top_books = top_books.sort_values('avg_score', ascending=False).head(20)
        
        # Análise de sentimento básica
        sentiment_analysis = {
            'positive_reviews': int(len(df[df['score'] >= 4])),
            'neutral_reviews': int(len(df[(df['score'] >= 3) & (df['score'] < 4)])),
            'negative_reviews': int(len(df[df['score'] < 3])),
            'total_reviews': len(df)
        }
        
        # Distribuição de scores
        score_distribution = df['score'].value_counts().sort_index().to_dict()
        
        results = {
            'top_books': top_books.reset_index().to_dict('records'),
            'sentiment_analysis': sentiment_analysis,
            'score_distribution': score_distribution,
            'data_summary': {
                'total_reviews': len(df),
                'unique_books': int(df['Id'].nunique()),
                'unique_users': int(df['User_id'].nunique()),
                'avg_score': float(df['score'].mean())
            }
        }
        
        return results
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Executa o pipeline completo de processamento.
        
        Returns:
            DataFrame processado
        """
        logger.info("Iniciando processamento otimizado de %s", self.input_file.name)
        
        # Análise inicial
        file_analysis = self.analyze_file_structure()
        
        # Processamento
        processed_df = self.process_data()
        
        if len(processed_df) > 0:
            # Salvamento
            self.save_processed_data(processed_df)
            
            logger.info("Processamento concluído com sucesso!")
            return processed_df
        else:
            logger.error("Falha no processamento - DataFrame vazio")
            return pd.DataFrame()

def main():
    """Função principal para execução do processador."""
    processor = OptimizedDataProcessor()
    
    try:
        processed_data = processor.run_full_pipeline()
        
        if len(processed_data) > 0:
            print("\n[OK] Processamento concluído!")
            print(f"[INFO] Total de registros processados: {len(processed_data)}")
            print(f"[INFO] Arquivos gerados em: data/")
        else:
            print("\n[ERRO] Falha no processamento")
            
    except Exception as e:
        logger.error(f"Erro durante o processamento: {e}")
        print(f"\n[ERRO] Falha no processamento: {e}")

if __name__ == "__main__":
    main() 