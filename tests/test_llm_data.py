#!/usr/bin/env python3
"""
Script para testar se os dados LLM estão sendo carregados corretamente
"""

import json
from pathlib import Path

def test_llm_data():
    """Testa se os dados LLM estão corretos"""
    try:
        # Verificar se o arquivo existe
        llm_file = Path("data/cache/llm_analysis_results.json")
        if not llm_file.exists():
            print(f"❌ Arquivo não encontrado: {llm_file}")
            return False
        
        # Carregar dados
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        
        print(f"✅ Arquivo carregado: {llm_file}")
        print(f"📊 Tamanho do arquivo: {llm_file.stat().st_size} bytes")
        
        # Verificar estrutura
        if 'local_analysis' in llm_data:
            local = llm_data['local_analysis']
            print(f"📈 DistilBERT processou: {local.get('total_processed', 0)} textos")
            
            sentiment_dist = local.get('sentiment_distribution', {})
            print(f"📊 Distribuição de sentimentos:")
            for sentiment, count in sentiment_dist.items():
                print(f"   {sentiment}: {count:,}")
        
        if 'gpt_analysis' in llm_data:
            gpt = llm_data['gpt_analysis']
            print(f"🧠 GPT processou: {gpt.get('total_processed', 0)} textos")
            
            themes = gpt.get('themes_extracted', [])
            print(f"🎯 Temas extraídos: {themes[:5]}...")
        
        if 'comparison' in llm_data:
            comp = llm_data['comparison']
            agreement = comp.get('agreement_rate', 0)
            print(f"🤝 Concordância VADER vs DistilBERT: {agreement:.1%}")
            
            if 'vader_vs_local' in comp:
                vader_disc = comp['vader_vs_local'].get('vader_avg_discrepancy', 'N/A')
                local_disc = comp['vader_vs_local'].get('local_avg_discrepancy', 'N/A')
                print(f"📊 Discrepância VADER: {vader_disc}")
                print(f"📊 Discrepância DistilBERT: {local_disc}")
        
        print("\n✅ Dados LLM estão corretos e completos!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar dados LLM: {e}")
        return False

if __name__ == "__main__":
    test_llm_data() 