"""
Módulo de análise LLM híbrida para complementar o VADER existente.
Estratégia: DistilBERT (rápido) + GPT (qualitativo) + comparação com VADER.
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

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridLLMAnalyzer:
    """
    Analisador LLM híbrido que complementa o VADER existente.
    """
    
    def __init__(self):
        """Inicializa o analisador híbrido."""
        load_dotenv()
        self.local_pipeline = None
        self.openai_client = None
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def _setup_local_pipeline(self):
        """Configura o pipeline local (DistilBERT)."""
        try:
            from transformers import pipeline
            import torch
            
            if self.local_pipeline is None:
                logger.info("Carregando modelo DistilBERT (modo otimizado)...")
                
                self.local_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,  # Sempre CPU para evitar problemas de GPU
                    return_all_scores=False,  # Apenas a label mais provável
                    truncation=True,
                    max_length=256,  # Reduzido para performance
                    padding=True,
                    batch_size=8
                )
                logger.info("Modelo DistilBERT carregado com sucesso!")
                    
        except ImportError:
            logger.warning("Transformers não disponível. Pulando análise local.")
            return False
        except Exception as e:
            logger.error(f"Erro ao carregar DistilBERT: {e}")
            return False
        return True
    
    def _setup_openai_client(self):
        """Configura o cliente OpenAI."""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY não encontrada. Pulando análise GPT.")
                return False
            
            self.openai_client = openai
            self.openai_client.api_key = api_key
            logger.info("Cliente OpenAI configurado!")
            return True
        except ImportError:
            logger.warning("OpenAI não disponível. Pulando análise GPT.")
            return False
    
    def analyze_sentiment_local(self, texts: List[str], batch_size: int = 16) -> List[str]:
        """
        Análise de sentimento local usando DistilBERT.
        
        Args:
            texts: Lista de textos para análise
            batch_size: Tamanho do lote para processamento
            
        Returns:
            Lista de sentimentos (POSITIVE/NEGATIVE)
        """
        if not self._setup_local_pipeline():
            return ['NEUTRAL'] * len(texts)
        
        logger.info(f"Analisando {len(texts)} textos com DistilBERT...")
        start_time = time.time()
        
        # Limitar para evitar travamentos - máximo 5000 textos
        if len(texts) > 5000:
            logger.warning(f"Limitando análise de {len(texts)} para 5000 textos para evitar travamentos")
            texts = texts[:5000]
        
        # Truncar textos de forma mais agressiva
        truncated_texts = []
        for text in texts:
            # Truncar para máximo 200 palavras (~150-200 tokens)
            words = str(text).split()[:200]
            safe_text = ' '.join(words)[:600]  # 600 chars máximo
            truncated_texts.append(safe_text)
        
        results = []
        total_batches = len(truncated_texts) // batch_size + (1 if len(truncated_texts) % batch_size > 0 else 0)
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Processar batch inteiro de uma vez (mais eficiente)
                batch_results = self.local_pipeline(batch)
                
                # Extrair labels dos resultados
                for result in batch_results:
                    if isinstance(result, dict) and 'label' in result:
                        results.append(result['label'])
                    elif isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'label' in result[0]:
                            results.append(result[0]['label'])
                        else:
                            results.append('NEUTRAL')
                    else:
                        results.append('NEUTRAL')
                        
            except Exception as e:
                logger.error(f"Erro no batch {batch_num}: {str(e)[:100]}")
                # Adicionar neutros para manter o mesmo número de resultados
                results.extend(['NEUTRAL'] * len(batch))
            
            # Log a cada 10 batches (mais frequente para debug)
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"   Processados {batch_num}/{total_batches} batches ({len(results)}/{len(truncated_texts)} textos)")
        
        elapsed = time.time() - start_time
        rate = len(results)/elapsed if elapsed > 0 else 0
        logger.info(f"Análise local concluída em {elapsed:.2f}s ({rate:.1f} textos/s)")
        
        return results
    
    def analyze_with_gpt(self, text: str, score: int) -> Dict[str, Any]:
        """
        Análise qualitativa com GPT.
        
        Args:
            text: Texto para análise
            score: Score do usuário (1-5)
            
        Returns:
            Dicionário com análise detalhada
        """
        if not self._setup_openai_client():
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'themes': [],
                'summary': 'Análise não disponível',
                'discrepancy': 'none'
            }
        
        prompt = f"""
        Analise esta resenha de livro (score do usuário: {score}/5):
        "{text[:500]}"  # Limita a 500 chars para economia
        
        Responda em JSON:
        {{
            "sentiment": "positive/negative/neutral",
            "confidence": 0.0-1.0,
            "themes": ["tema1", "tema2"],
            "summary": "resumo em 1 frase",
            "discrepancy": "alta/baixa/none" (se score não combina com texto)
        }}
        """
        
        try:
            # API OpenAI v1.0+
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise GPT: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'themes': [],
                'summary': 'Erro na análise',
                'discrepancy': 'none'
            }
    
    def compare_sentiment_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compara os diferentes métodos de análise de sentimento.
        
        Args:
            df: DataFrame com colunas 'sentiment' (VADER), 'sentiment_local' (DistilBERT), 'llm_analysis' (GPT)
            
        Returns:
            Dicionário com comparações
        """
        logger.info("Comparando métodos de análise de sentimento...")
        
        # Preparar dados para comparação
        comparison_data = df.dropna(subset=['sentiment', 'sentiment_local']).copy()
        
        # Mapear sentimentos para valores numéricos (maiúsculas e minúsculas)
        sentiment_map = {
            'positive': 1, 'neutral': 0, 'negative': -1,
            'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1,
            'pos': 1, 'neu': 0, 'neg': -1
        }
        
        # Limpar dados de sentimento antes do mapeamento
        comparison_data['sentiment_clean'] = comparison_data['sentiment'].astype(str).str.lower()
        comparison_data['sentiment_local_clean'] = comparison_data['sentiment_local'].astype(str).str.lower()
        
        comparison_data['vader_numeric'] = comparison_data['sentiment_clean'].map(sentiment_map)
        comparison_data['local_numeric'] = comparison_data['sentiment_local_clean'].map(sentiment_map)
        
        # Remover NaNs do mapeamento
        comparison_data = comparison_data.dropna(subset=['vader_numeric', 'local_numeric'])
        
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
        
        # Análise GPT (se disponível)
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
                f"Concordância VADER vs DistilBERT: {agreement:.1%}",
                f"Discrepância média VADER vs Score: {comparison_data['vader_discrepancy'].mean():.2f}",
                f"Discrepância média DistilBERT vs Score: {comparison_data['local_discrepancy'].mean():.2f}"
            ]
        }
        
        if gpt_analysis:
            results['insights'].extend([
                f"Análise GPT: {gpt_analysis['total_analyzed']} textos processados",
                f"Temas mais frequentes: {', '.join(gpt_analysis['themes_extracted'][:3])}",
                f"Discrepâncias detectadas: {gpt_analysis['discrepancies_found']}"
            ])
        
        return results
    
    def _extract_common_themes(self, llm_results: pd.Series) -> List[str]:
        """Extrai temas comuns dos resultados GPT."""
        all_themes = []
        for result in llm_results:
            if isinstance(result, dict) and 'themes' in result:
                all_themes.extend(result['themes'])
        
        if all_themes:
            theme_counts = pd.Series(all_themes).value_counts()
            return theme_counts.head(10).index.tolist()
        return []
    
    def _count_discrepancies(self, llm_results: pd.Series) -> int:
        """Conta discrepâncias detectadas pelo GPT."""
        count = 0
        for result in llm_results:
            if isinstance(result, dict) and result.get('discrepancy') in ['alta', 'baixa']:
                count += 1
        return count
    
    def run_hybrid_analysis(self, df: pd.DataFrame, gpt_sample_size: int = 500) -> Dict[str, Any]:
        """
        Executa análise híbrida completa.
        
        Args:
            df: DataFrame com dados processados
            gpt_sample_size: Tamanho da amostra para análise GPT
            
        Returns:
            Resultados da análise híbrida
        """
        logger.info("🚀 Iniciando análise LLM híbrida...")
        
        # Verificar se já temos análise VADER
        if 'sentiment' not in df.columns:
            logger.warning("Coluna 'sentiment' (VADER) não encontrada!")
            return {'error': 'Análise VADER não encontrada'}
        
        # 1. Análise Local (DistilBERT) - amostra limitada para evitar travamentos
        logger.info("🔍 Camada 1: Análise local (DistilBERT)")
        vader_data = df.dropna(subset=['sentiment'])
        
        # Limitar para 3000 textos máximo para evitar travamentos
        if len(vader_data) > 3000:
            logger.info(f"Limitando análise DistilBERT de {len(vader_data)} para 3000 textos")
            vader_sample = vader_data.sample(n=3000, random_state=42)
        else:
            vader_sample = vader_data
            
        vader_texts = vader_sample['text'].tolist()
        local_sentiments = self.analyze_sentiment_local(vader_texts)
        
        # Adicionar ao DataFrame (apenas para a amostra processada)
        df.loc[vader_sample.index, 'sentiment_local'] = local_sentiments
        
        # 2. Análise GPT (Qualitativa) - amostra menor
        logger.info(f"🧠 Camada 2: Análise GPT (amostra de {gpt_sample_size})")
        gpt_sample = df.dropna(subset=['sentiment']).sample(
            n=min(gpt_sample_size, len(df.dropna(subset=['sentiment']))), 
            random_state=42
        )
        
        gpt_results = []
        for idx, row in gpt_sample.iterrows():
            if idx % 50 == 0:
                logger.info(f"   Processando GPT {idx}/{len(gpt_sample)}...")
            
            result = self.analyze_with_gpt(row['text'], row['score'])
            gpt_results.append(result)
        
        gpt_sample = gpt_sample.copy()
        gpt_sample['llm_analysis'] = gpt_results
        
        # 3. Comparação entre métodos
        logger.info("📊 Comparando métodos de análise...")
        comparison = self.compare_sentiment_methods(df)
        
        # 4. Salvar resultados
        results = {
            'local_analysis': {
                'total_processed': len(vader_texts),
                'sentiment_distribution': df['sentiment_local'].value_counts().to_dict(),
                'processing_time': '~30 segundos'
            },
            'gpt_analysis': {
                'total_processed': len(gpt_sample),
                'themes_extracted': self._extract_common_themes(gpt_results),
                'discrepancies_found': self._count_discrepancies(gpt_results),
                'avg_confidence': np.mean([r.get('confidence', 0.5) for r in gpt_results]),
                'processing_time': '~5 minutos'
            },
            'comparison': comparison,
            'insights': comparison['insights']
        }
        
        # Salvar cache
        cache_file = self.cache_dir / "llm_analysis_results.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("✅ Análise LLM híbrida concluída!")
        return results

def main():
    """Função principal para teste do analisador."""
    analyzer = HybridLLMAnalyzer()
    
    # Carregar dados processados
    try:
        df = pd.read_csv("data/merged_data_clean.csv")
        logger.info(f"Dados carregados: {len(df)} registros")
        
        # Executar análise híbrida
        results = analyzer.run_hybrid_analysis(df, gpt_sample_size=200)
        
        print("\n=== RESULTADOS DA ANÁLISE LLM HÍBRIDA ===")
        print(f"📊 Total processado (local): {results['local_analysis']['total_processed']}")
        print(f"🧠 Total processado (GPT): {results['gpt_analysis']['total_processed']}")
        print(f"📈 Concordância VADER vs DistilBERT: {results['comparison']['agreement_rate']:.1%}")
        
        if results['gpt_analysis']['themes_extracted']:
            print(f"🎯 Temas mais frequentes: {', '.join(results['gpt_analysis']['themes_extracted'][:3])}")
        
        print("\n✅ Análise concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante análise: {e}")
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    main()
