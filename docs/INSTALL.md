# 📋 Instalação

## 🚀 Instalação Rápida

### Pré-requisitos
- **Python 3.8+** 
- **4GB RAM** mínimo
- **2GB espaço** em disco

### Instalação
```bash
# 1. Clone o repositório
git clone <repository-url>
cd A3

# 2. Crie ambiente virtual
python -m venv a3env

# 3. Ative o ambiente
# Windows:
a3env\Scripts\activate
# Linux/Mac:
source a3env/bin/activate

# 4. Instale dependências
pip install -r requirements.txt
```

### Configuração Opcional
```bash
# Para análise LLM com OpenAI
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
```

## ✅ Teste da Instalação

```bash
# Teste básico
python -c "import pandas, streamlit, plotly; print('✅ Instalação OK')"

# Teste completo
python executar_solucao.py
```

## 🔧 Problemas Comuns

### Erro de Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Erro de Memória
```bash
# Ajuste configurações se necessário
export PYTHONHASHSEED=0
```

**🎯 Instalação concluída! Execute: `python executar_solucao.py`** 