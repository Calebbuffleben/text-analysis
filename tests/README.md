# Testes do Text Analysis Service

Este diretório contém testes para o serviço de análise de texto.

## Estrutura

- `test_sales_category_classification.py`: Testes para classificação de categorias de vendas
- `conftest.py`: Configuração compartilhada para pytest

## Executar Testes

### Instalar dependências de teste

```bash
pip install -r requirements.txt
```

### Executar todos os testes

```bash
# Na raiz do projeto text-analysis
pytest tests/

# Com mais verbosidade
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

### Executar testes específicos

```bash
# Apenas testes de classificação de categorias
pytest tests/test_sales_category_classification.py

# Apenas testes unitários
pytest tests/test_sales_category_classification.py::TestSalesCategoryClassification

# Apenas testes de integração
pytest tests/test_sales_category_classification.py::TestSalesCategoryIntegration
```

## Notas

- Os testes usam mocks para evitar carregar modelos ML reais (mais rápido)
- Testes de integração podem ser mais lentos se executarem modelos reais
- Para testes com modelos reais, configure variáveis de ambiente apropriadas

