#!/bin/bash
# Script de setup para rodar text-analysis localmente (fora do Docker)

set -e

echo "🚀 Configurando text-analysis para execução local..."

# Verificar Python
if ! command -v python3.11 &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.11 ou superior não encontrado. Por favor, instale Python primeiro."
    exit 1
fi

# Usar python3.11 se disponível, caso contrário python3
PYTHON_CMD=$(command -v python3.11 || command -v python3)

echo "✅ Usando Python: $PYTHON_CMD"

# Criar ambiente virtual
echo "📦 Criando ambiente virtual..."
$PYTHON_CMD -m venv venv

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Atualizar pip
echo "⬆️  Atualizando pip..."
pip install --upgrade pip

# Instalar dependências
echo "📚 Instalando dependências Python..."
pip install -r requirements.txt

# Baixar modelo spaCy
echo "🌐 Baixando modelo spaCy em português..."
python -m spacy download pt_core_news_sm

echo ""
echo "✅ Setup concluído com sucesso!"
echo ""
echo "Para executar o serviço:"
echo "  1. cd apps/text-analysis"
echo "  2. source venv/bin/activate"
echo "  3. python -m src.main"
echo ""
echo "Para testar:"
echo "  curl http://localhost:8000/health"
echo ""

