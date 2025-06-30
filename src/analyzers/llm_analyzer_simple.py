"""
Versão simplificada do analisador LLM para demonstração.
Funciona sem dependências externas (transformers, OpenAI).
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
import random

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLLMAnalyzer:
    """
    Analisador LLM simplificado para demonstração.
    Simula análise local e GPT sem dependências externas.
    """
    
    def __init__(self):
        """Inicializa o analisador simplificado."""
        load_dotenv()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Palavras-chave para análise local simulada
        self.positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'enjoy', 'fantastic', 'brilliant', 'outstanding']
        self.negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointing', 'boring', 'waste', 'poor', 'worst', 'horrible']
        
    def analyze_sentiment_local_simulated(self, texts: List[str]) -> List[str]:
        """
        Análise de sentimento local simulada (sem transformers).
        
        Args:
            texts: Lista de textos para análise
            
        Returns:
            Lista de sentimentos (POSITIVE/NEGATIVE/NEUTRAL)
        """
        logger.info(f"Simulando análise local para {len(texts)} textos...")
        start_time = time.time()
        
        results = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"   Processados {i}/{len(texts)} textos...")
            
            # Análise simples baseada em palavras-chave
            text_lower = text.lower()
            pos_count = sum(1 for word in self.positive_words if word in text_lower)
            neg_count = sum(1 for word in self.negative_words if word in text_lower)
            
            if pos_count > neg_count:
                results.append('POSITIVE')
            elif neg_count > pos_count:
                results.append('NEGATIVE')
            else:
                results.append('NEUTRAL')
        
        elapsed = time.time() - start_time
        logger.info(f"Análise local simulada concluída em {elapsed:.2f}s")
        
        return results
    
    def analyze_with_gpt_simulated(self, text: str, score: int) -> Dict[str, Any]:
        """
        Análise GPT simulada (sem API OpenAI).
        
        Args:
            text: Texto para análise
            score: Score do usuário (1-5)
            
        Returns:
            Dicionário com análise simulada
        """
        # Simular análise baseada no score e texto
        text_lower = text.lower()
        
        # Sentimento baseado em palavras-chave
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
        elif neg_count > pos_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Temas simulados baseados em palavras-chave
        themes = []
        if any(word in text_lower for word in ['character', 'characters', 'protagonist']):
            themes.append('personagens')
        if any(word in text_lower for word in ['plot', 'story', 'narrative']):
            themes.append('narrativa')
        if any(word in text_lower for word in ['writing', 'style', 'prose']):
            themes.append('estilo')
        if any(word in text_lower for word in ['ending', 'conclusion', 'final']):
            themes.append('final')
        if any(word in text_lower for word in ['beginning', 'start', 'opening']):
            themes.append('início')
        
        if not themes:
            themes = ['geral']
        
        # Resumo simulado
        if sentiment == 'positive':
            summary = "Avaliação positiva do livro"
        elif sentiment == 'negative':
            summary = "Avaliação negativa do livro"
        else:
            summary = "Avaliação neutra do livro"
        
        # Discrepância simulada
        score_sentiment = 'positive' if score >= 4 else ('negative' if score <= 2 else 'neutral')
        if sentiment != score_sentiment:
            discrepancy = 'alta' if abs(score - 3) > 1 else 'baixa'
        else:
            discrepancy = 'none'
        
        return {
            'sentiment': sentiment,
            'confidence': random.uniform(0.7, 0.95),
            'themes': themes,
            'summary': summary,
            'discrepancy': discrepancy
        }
    
    def compare_sentiment_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compara os diferentes métodos de análise de sentimento.
        
        Args:
            df: DataFrame com colunas 'sentiment' (VADER), 'sentiment_local' (Simulado), 'llm_analysis' (GPT Simulado)
            
        Returns:
            Dicionário com comparações
        """
        logger.info("Comparando métodos de análise de sentimento...")
        
        # Preparar dados para comparação
        comparison_data = df.dropna(subset=['sentiment', 'sentiment_local']).copy()
        
        # Mapear sentimentos para valores numéricos
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        comparison_data['vader_numeric'] = comparison_data['sentiment'].map(sentiment_map)
        comparison_data['local_numeric'] = comparison_data['sentiment_local'].map(sentiment_map)
        
        # Calcular concordância
        agreement = (comparison_data['vader_numeric'] == comparison_data['local_numeric']).mean()
        
        # Análise de discrepâncias com score
        def calculate_discrepancy(row):
            score_sentiment = 1 if row['score'] >= 4 else (-1 if row['score'] <= 2 else 0)
            vader_discrepancy = abs(score_sentiment - row['vader_numeric'])
            local_discrepancy = abs(score_sentiment - row['local_numeric'])
            return vader_discrepancy, local_discrepancy
        
        comparison_data[['vader_discrepancy', 'local_discrepancy']] = comparison_data.apply(
            calculate_discrepancy, axis=1, result_type='expand'
        )
        
        # Análise GPT simulada (se disponível)
        gpt_analysis = None
        if 'llm_analysis' in df.columns:
            gpt_data = df.dropna(subset=['llm_analysis'])
            if len(gpt_data) > 0:
                gpt_analysis = {
                    'total_analyzed': len(gpt_data),
                    'themes_extracted': self._extract_common_themes(gpt_data['llm_analysis']),
                    'discrepancies_found': self._count_discrepancies(gpt_data['llm_analysis']),
                    'avg_confidence': np.mean([r.get('confidence', 0.5) for r in gpt_data['llm_analysis']])
                }
        
        results = {
            'agreement_rate': float(agreement),
            'total_comparison': len(comparison_data),
            'vader_vs_local': {
                'agreement': float(agreement),
                'vader_avg_discrepancy': float(comparison_data['vader_discrepancy'].mean()),
                'local_avg_discrepancy': float(comparison_data['local_discrepancy'].mean())
            },
            'gpt_analysis': gpt_analysis,
            'insights': [
                f"Concordância VADER vs Simulado: {agreement:.1%}",
                f"Discrepância média VADER vs Score: {comparison_data['vader_discrepancy'].mean():.2f}",
                f"Discrepância média Simulado vs Score: {comparison_data['local_discrepancy'].mean():.2f}",
                "[AVISO] Análise simulada para demonstração (sem dependências externas)"
            ]
        }
        
        if gpt_analysis:
            results['insights'].extend([
                f"Análise GPT Simulada: {gpt_analysis['total_analyzed']} textos processados",
                f"Temas mais frequentes: {', '.join(gpt_analysis['themes_extracted'][:3])}",
                f"Discrepâncias detectadas: {gpt_analysis['discrepancies_found']}"
            ])
        
        return results
    
    def _extract_common_themes(self, llm_results: pd.Series) -> List[str]:
        """Extrai temas comuns dos resultados GPT simulados."""
        all_themes = []
        for result in llm_results:
            if isinstance(result, dict) and 'themes' in result:
                all_themes.extend(result['themes'])
        
        if all_themes:
            theme_counts = pd.Series(all_themes).value_counts()
            return theme_counts.head(10).index.tolist()
        return []
    
    def _count_discrepancies(self, llm_results: pd.Series) -> int:
        """Conta discrepâncias detectadas pelo GPT simulado."""
        count = 0
        for result in llm_results:
            if isinstance(result, dict) and result.get('discrepancy') in ['alta', 'baixa']:
                count += 1
        return count
    
    def run_hybrid_analysis(self, df: pd.DataFrame, gpt_sample_size: int = 500) -> Dict[str, Any]:
        """
        Executa análise híbrida completa (simulada).
        
        Args:
            df: DataFrame com dados processados
            gpt_sample_size: Tamanho da amostra para análise GPT
            
        Returns:
            Resultados da análise híbrida
        """
        logger.info("Iniciando análise LLM híbrida (simulada)...")
        
        # Verificar se já temos análise VADER
        if 'sentiment' not in df.columns:
            logger.warning("Coluna 'sentiment' (VADER) não encontrada!")
            return {'error': 'Análise VADER não encontrada'}
        
        # 1. Análise Local Simulada - todos os textos com VADER
        logger.info("Camada 1: Análise local simulada")
        vader_texts = df.dropna(subset=['sentiment'])['text'].tolist()
        local_sentiments = self.analyze_sentiment_local_simulated(vader_texts)
        
        # Adicionar ao DataFrame
        vader_mask = df['sentiment'].notna()
        df.loc[vader_mask, 'sentiment_local'] = local_sentiments
        
        # 2. Análise GPT Simulada - amostra menor
        logger.info(f"Camada 2: Análise GPT simulada (amostra de {gpt_sample_size})")
        gpt_sample = df.dropna(subset=['sentiment']).sample(
            n=min(gpt_sample_size, len(df.dropna(subset=['sentiment']))), 
            random_state=42
        )
        
        gpt_results = []
        for idx, row in gpt_sample.iterrows():
            if idx % 50 == 0:
                logger.info(f"   Processando GPT simulado {idx}/{len(gpt_sample)}...")
            
            result = self.analyze_with_gpt_simulated(row['text'], row['score'])
            gpt_results.append(result)
        
        gpt_sample = gpt_sample.copy()
        gpt_sample['llm_analysis'] = gpt_results
        
        # 3. Comparação entre métodos
        logger.info("Comparando métodos de análise...")
        comparison = self.compare_sentiment_methods(df)
        
        # 4. Salvar resultados
        results = {
            'local_analysis': {
                'total_processed': len(vader_texts),
                'sentiment_distribution': df['sentiment_local'].value_counts().to_dict(),
                'processing_time': '~5 segundos (simulado)'
            },
            'gpt_analysis': {
                'total_processed': len(gpt_sample),
                'themes_extracted': self._extract_common_themes(gpt_results),
                'discrepancies_found': self._count_discrepancies(gpt_results),
                'avg_confidence': np.mean([r.get('confidence', 0.5) for r in gpt_results]),
                'processing_time': '~10 segundos (simulado)'
            },
            'comparison': comparison,
            'insights': comparison['insights']
        }
        
        # Salvar cache
        cache_file = self.cache_dir / "llm_analysis_results.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("Análise LLM híbrida (simulada) concluída!")
        return results

def main():
    """Função principal para teste do analisador simplificado."""
    analyzer = SimpleLLMAnalyzer()
    
    # Carregar dados processados
    try:
        df = pd.read_csv("data/merged_data_clean.csv")
        logger.info(f"Dados carregados: {len(df)} registros")
        
        # Executar análise híbrida
        results = analyzer.run_hybrid_analysis(df, gpt_sample_size=200)
        
        print("\n=== RESULTADOS DA ANÁLISE LLM HÍBRIDA (SIMULADA) ===")
        print(f"[+] Total processado (local): {results['local_analysis']['total_processed']}")
        print(f"[+] Total processado (GPT): {results['gpt_analysis']['total_processed']}")
        print(f"[+] Concordância VADER vs Simulado: {results['comparison']['agreement_rate']:.1%}")
        
        if results['gpt_analysis']['themes_extracted']:
            print(f"[+] Temas mais frequentes: {', '.join(results['gpt_analysis']['themes_extracted'][:3])}")
        
        print("\n[OK] Análise simulada concluída com sucesso!")
        print("[INFO] Esta é uma versão de demonstração sem dependências externas")
        
    except Exception as e:
        logger.error(f"Erro durante análise: {e}")
        print(f"\n[ERRO] Erro: {e}")

if __name__ == "__main__":
    main()
