"""
Script de teste para verificar a integração LLM.
"""

import json
import pandas as pd
from pathlib import Path

def test_llm_integration():
    """Testa a integração LLM."""
    print("🧪 Testando integração LLM...")
    
    # 1. Verificar se o módulo LLM existe
    try:
        from llm_analyzer import HybridLLMAnalyzer
        print("✅ Módulo LLM carregado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao carregar módulo LLM: {e}")
        return False
    
    # 2. Verificar se os dados processados existem
    data_file = Path("data/merged_data_clean.csv")
    if not data_file.exists():
        print("❌ Arquivo de dados processados não encontrado")
        print("   Execute primeiro: python optimized_processor.py")
        return False
    
    print("✅ Dados processados encontrados")
    
    # 3. Verificar se há análise VADER
    try:
        df = pd.read_csv(data_file)
        if 'sentiment' not in df.columns:
            print("❌ Coluna 'sentiment' (VADER) não encontrada")
            print("   Execute primeiro: python optimized_processor.py")
            return False
        
        vader_count = df['sentiment'].notna().sum()
        print(f"✅ Análise VADER encontrada: {vader_count:,} textos")
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return False
    
    # 4. Verificar se há resultados LLM
    llm_cache = Path("data/cache/llm_analysis_results.json")
    if llm_cache.exists():
        try:
            with open(llm_cache, 'r', encoding='utf-8') as f:
                llm_results = json.load(f)
            
            if 'error' not in llm_results:
                print("✅ Resultados LLM encontrados")
                print(f"   - Local: {llm_results['local_analysis']['total_processed']:,} textos")
                print(f"   - GPT: {llm_results['gpt_analysis']['total_processed']:,} textos")
                print(f"   - Concordância: {llm_results['comparison']['agreement_rate']:.1%}")
                return True
            else:
                print(f"⚠️  Erro nos resultados LLM: {llm_results['error']}")
        except Exception as e:
            print(f"❌ Erro ao carregar resultados LLM: {e}")
    else:
        print("⚠️  Resultados LLM não encontrados")
        print("   Execute: python llm_analyzer.py")
    
    return False

def test_workflow_integration():
    """Testa a integração no workflow."""
    print("\n🔧 Testando integração no workflow...")
    
    try:
        from workflow_analysis import OptimizedAnalysisWorkflow
        
        workflow = OptimizedAnalysisWorkflow()
        
        # Testar análise LLM
        results = workflow.analyze_llm_hybrid(50)  # Amostra pequena para teste
        
        if 'error' not in results:
            print("✅ Integração no workflow funcionando")
            return True
        else:
            print(f"⚠️  Erro na integração: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste do workflow: {e}")
        return False

def main():
    """Função principal."""
    print("=================================================================================")
    print("🧪 TESTE DE INTEGRAÇÃO LLM")
    print("=================================================================================")
    
    # Teste básico
    llm_ok = test_llm_integration()
    
    # Teste workflow
    workflow_ok = test_workflow_integration()
    
    print("\n=================================================================================")
    print("📊 RESUMO DOS TESTES")
    print("=================================================================================")
    
    if llm_ok and workflow_ok:
        print("✅ TODOS OS TESTES PASSARAM!")
        print("🎉 Integração LLM funcionando perfeitamente")
    elif llm_ok:
        print("⚠️  Módulo LLM OK, mas workflow com problemas")
    elif workflow_ok:
        print("⚠️  Workflow OK, mas módulo LLM com problemas")
    else:
        print("❌ PROBLEMAS ENCONTRADOS")
        print("🔧 Verifique:")
        print("   1. Dependências instaladas: pip install -r requirements.txt")
        print("   2. Dados processados: python optimized_processor.py")
        print("   3. Análise LLM: python llm_analyzer.py")
        print("   4. Chave OpenAI no .env (opcional)")

if __name__ == "__main__":
    main() 