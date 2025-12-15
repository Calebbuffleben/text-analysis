"""
Configuração compartilhada para testes pytest.

Este arquivo contém fixtures e configurações que são compartilhadas
entre todos os testes do projeto.
"""

import pytest
import sys
from pathlib import Path

# Adicionar diretório raiz ao path para imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Configurar pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

