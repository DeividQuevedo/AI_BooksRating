#!/usr/bin/env python3
"""
Script principal para executar a solução de análise de livros.
Organização profissional - chama o script principal da pasta src/main/
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path para imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    try:
        # Importar e executar o script principal
        from src.main.executar_solucao import main
        success = main()
        if not success:
            sys.exit(1)
    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        print("💡 Verifique se todos os arquivos estão nas pastas corretas")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        sys.exit(1) 