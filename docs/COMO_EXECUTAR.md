# 🚀 Como Executar o Projeto

## ⚡ Execução Rápida

### 1. Configuração Inicial
```bash
# Clone e configure o ambiente
git clone <repository-url> && cd A3
python -m venv a3env
a3env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Executar Pipeline Completo
```bash
python executar_solucao.py
```

**Este comando executa:**
- ✅ Processamento de dados (2 min)
- ✅ Análise inteligente (30s) 
- ✅ Dashboard interativo (automático)

### 3. Acessar Dashboard
**URL:** http://localhost:8501

---

## 🧠 Análise LLM (Opcional)

### Executar Análise LLM Real
```bash
python llm_analyzer.py
```
**Resultado:** Análise com DistilBERT + OpenAI GPT

### Configurar OpenAI (Opcional)
```bash
# Criar arquivo .env
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
```

---

## 📊 Execução por Etapas

### Processamento Individual
```bash
# Apenas processamento
python optimized_processor.py

# Apenas análise
python workflow_analysis.py

# Apenas dashboard
streamlit run streamlit_app/app.py
```

---

## 🔧 Solução de Problemas

### Erro de Memória
```bash
# Use menos dados de demonstração
python executar_solucao.py --demo
```

### Erro de Dependências
```bash
# Reinstale dependências
pip install --upgrade -r requirements.txt
```

### Dashboard não Carrega
- Verifique se está em http://localhost:8501
- Tente http://localhost:8502 ou 8503

---

## ✅ Status Esperado

**Sucesso quando ver:**
```
🎯 PIPELINE CONCLUÍDO
📊 Dados processados: 25.002 registros
🧠 Análise LLM: CONCLUÍDA
📈 Dashboard: http://localhost:8501
```

**🎉 Projeto pronto para demonstração!** 