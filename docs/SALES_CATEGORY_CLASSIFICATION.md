# Classificação de Categorias de Vendas com SBERT

## Visão Geral

O sistema de classificação de categorias de vendas utiliza análise semântica com SBERT (Sentence-BERT) para identificar automaticamente o estágio da conversa de vendas. Esta funcionalidade permite detectar quando o cliente demonstra interesse em preço, explora valor, apresenta objeções, está pronto para fechar, entre outros sinais importantes.

## Como Funciona

### Arquitetura

1. **Exemplos de Referência**: Cada categoria possui 10 exemplos de texto representativos em português
2. **Embeddings Pré-calculados**: Os embeddings dos exemplos são calculados uma vez e armazenados em cache
3. **Classificação Semântica**: O texto de entrada é comparado semanticamente com os exemplos usando similaridade de cosseno
4. **Seleção da Categoria**: A categoria com maior similaridade média é selecionada
5. **Cálculo de Confiança**: A confiança é calculada baseada na diferença entre melhor e segunda melhor categoria

### Fluxo de Dados

```
Texto de Entrada
    ↓
Geração de Embedding (SBERT)
    ↓
Comparação com Embeddings dos Exemplos
    ↓
Cálculo de Similaridade por Categoria
    ↓
Seleção da Categoria com Maior Score
    ↓
Cálculo de Confiança
    ↓
Retorno: (categoria, confiança, scores)
```

## Categorias Disponíveis

### 1. `price_interest`
**Descrição**: Cliente demonstra interesse explícito em saber o preço

**Exemplos**:
- "Quanto custa isso?"
- "Qual é o preço?"
- "Preciso saber o valor"

**Quando usar**: Identificar quando o cliente está considerando compra e precisa saber investimento

---

### 2. `value_exploration`
**Descrição**: Cliente explora o valor e benefícios da solução

**Exemplos**:
- "Como isso vai me ajudar?"
- "Qual o benefício disso para mim?"
- "Por que isso é melhor que outras opções?"

**Quando usar**: Identificar quando cliente está avaliando valor, não apenas preço

---

### 3. `objection_soft`
**Descrição**: Objeções leves, dúvidas ou hesitações não definitivas

**Exemplos**:
- "Não tenho certeza se preciso disso"
- "Preciso pensar melhor"
- "Talvez depois eu considere"

**Quando usar**: Identificar hesitações que podem ser resolvidas com mais informações

---

### 4. `objection_hard`
**Descrição**: Objeções fortes e definitivas, rejeição clara

**Exemplos**:
- "Não estou interessado"
- "Não preciso disso"
- "Muito caro para mim"

**Quando usar**: Identificar rejeições claras que requerem abordagem diferente

---

### 5. `decision_signal`
**Descrição**: Sinais claros de que o cliente está pronto para tomar decisão

**Exemplos**:
- "Quando posso começar?"
- "Como faço para contratar?"
- "Vamos fechar o negócio"

**Quando usar**: Identificar momento crítico para fechamento

---

### 6. `information_gathering`
**Descrição**: Cliente busca informações adicionais sobre a solução

**Exemplos**:
- "Me explique mais sobre isso"
- "Como funciona exatamente?"
- "Quais são as opções disponíveis?"

**Quando usar**: Identificar quando cliente precisa de mais detalhes técnicos

---

### 7. `stalling`
**Descrição**: Cliente está protelando ou adiando a decisão

**Exemplos**:
- "Deixa eu ver"
- "Vou pensar sobre isso"
- "Preciso consultar minha equipe"

**Quando usar**: Identificar procrastinação que pode precisar de urgência

---

### 8. `closing_readiness`
**Descrição**: Cliente demonstra prontidão para fechar o negócio

**Exemplos**:
- "Estou pronto para fechar"
- "Vamos fazer isso acontecer"
- "Quero avançar com isso"

**Quando usar**: Identificar prontidão máxima para fechamento imediato

## Configuração

### Variáveis de Ambiente

```bash
# SBERT Model (obrigatório para classificação de vendas)
SBERT_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Modelo BERT para sentimento (obrigatório)
MODEL_NAME=neuralmind/bert-base-portuguese-cased

# Device (cpu ou cuda)
MODEL_DEVICE=cpu

# Cache de modelos
MODEL_CACHE_DIR=/app/models/.cache
```

### Requisitos

- Python 3.11+
- PyTorch
- sentence-transformers >= 2.3.0
- transformers >= 4.37.2

## Uso

### Via API REST

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Quanto custa isso?",
    "meetingId": "meet_123",
    "participantId": "user_456",
    "timestamp": 1234567890
  }'
```

**Resposta**:
```json
{
  "meetingId": "meet_123",
  "participantId": "user_456",
  "text": "Quanto custa isso?",
  "analysis": {
    "intent": "ask_price",
    "intent_confidence": 0.8,
    "topic": "pricing",
    "topic_confidence": 0.9,
    "speech_act": "question",
    "speech_act_confidence": 0.9,
    "keywords": ["quanto", "custa"],
    "entities": ["preço"],
    "sentiment": "neutral",
    "sentiment_score": 0.5,
    "urgency": 0.65,
    "embedding": [0.123, 0.456, ...],
    "sales_category": "price_interest",
    "sales_category_confidence": 0.85
  },
  "timestamp": 1234567890,
  "confidence": 0.9
}
```

### Via Socket.IO

O serviço Python automaticamente classifica categorias de vendas quando recebe eventos `transcription_chunk` ou `audio_chunk` e retorna via `text_analysis_result`.

### Via Código Python

```python
from src.models.bert_analyzer import BERTAnalyzer

# Inicializar analisador
analyzer = BERTAnalyzer(
    model_name='neuralmind/bert-base-portuguese-cased',
    sbert_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Classificar texto
categoria, confianca, scores = analyzer.classify_sales_category(
    "Quanto custa isso?",
    min_confidence=0.3
)

print(f"Categoria: {categoria}")
print(f"Confiança: {confianca:.2%}")
print(f"Scores: {scores}")
```

### Script de Validação Manual

```bash
python scripts/validate_sales_category.py "Quanto custa isso?"
```

## Performance

### Latência

- **Primeira classificação**: ~400-500ms (carrega embeddings dos exemplos)
- **Classificações subsequentes**: ~5ms (usa cache)
- **Memória**: ~30KB (8 categorias × 10 exemplos × 384 dims)

### Otimizações

- **Cache de embeddings**: Embeddings dos exemplos são pré-calculados uma vez
- **Lazy loading**: Modelos são carregados apenas quando necessário
- **Normalização**: Embeddings normalizados para comparações eficientes

## Threshold de Confiança

O parâmetro `min_confidence` controla o score mínimo necessário para aceitar uma classificação:

- **0.3 (padrão)**: Permissivo, aceita classificações mais fracas
- **0.5**: Moderado, balanceado
- **0.7**: Restritivo, apenas classificações muito claras

Se o melhor score for menor que `min_confidence`, a função retorna `None` como categoria.

## Interpretação dos Scores

### Score de Categoria (0.0 a 1.0)

- **0.7-1.0**: Alta similaridade semântica com a categoria
- **0.5-0.7**: Similaridade moderada
- **0.3-0.5**: Similaridade baixa
- **0.0-0.3**: Muito baixa ou nenhuma similaridade

### Confiança (0.0 a 1.0)

- **0.7-1.0**: Alta confiança (diferença clara entre melhor e segunda melhor)
- **0.5-0.7**: Confiança moderada
- **0.3-0.5**: Confiança baixa (categorias muito próximas)
- **0.0-0.3**: Muito baixa confiança (classificação incerta)

## Casos de Uso no Backend

### Detecção de Estágio da Conversa

```typescript
if (textAnalysis.sales_category === 'price_interest') {
  // Cliente está interessado em preço - preparar proposta
}

if (textAnalysis.sales_category === 'decision_signal') {
  // Cliente está pronto - acelerar fechamento
}

if (textAnalysis.sales_category === 'objection_hard') {
  // Objeção forte - requer abordagem diferente
}
```

### Análise de Tendências

```typescript
// Rastrear mudanças de categoria ao longo da conversa
const categoryHistory = participantState.textAnalysisHistory.map(
  ta => ta.sales_category
);

// Detectar progressão: value_exploration → price_interest → decision_signal
```

## Troubleshooting

### Problema: `sales_category` sempre retorna `None`

**Possíveis causas**:
1. SBERT não está configurado (`SBERT_MODEL_NAME` não definido)
2. Score abaixo do threshold mínimo (`min_confidence` muito alto)
3. Texto muito ambíguo ou não relacionado a vendas

**Solução**:
- Verificar `Config.SBERT_MODEL_NAME`
- Reduzir `min_confidence` para 0.3 ou menos
- Verificar scores de todas as categorias no log

### Problema: Classificação incorreta

**Possíveis causas**:
1. Texto muito curto ou ambíguo
2. Exemplos de referência não cobrem variação linguística
3. Threshold muito baixo permitindo falsos positivos

**Solução**:
- Verificar confiança da classificação (deve ser > 0.5)
- Adicionar mais exemplos à categoria relevante
- Aumentar `min_confidence` para reduzir falsos positivos

### Problema: Performance lenta

**Possíveis causas**:
1. Primeira chamada (carrega embeddings)
2. Modelo rodando em CPU ao invés de GPU
3. Cache não está funcionando

**Solução**:
- Primeira chamada é esperada (~400ms)
- Configurar `MODEL_DEVICE=cuda` se GPU disponível
- Verificar que `_sales_examples_loaded` está True após primeira chamada

## Logs e Debugging

### Logs no Python

```python
# Log quando categoria é detectada
[INFO] ✅ [ANÁLISE] Categoria de vendas classificada
  sales_category=price_interest
  sales_category_confidence=0.85
  best_score=0.92

# Log quando nenhuma categoria detectada
[DEBUG] ⚠️ [ANÁLISE] Nenhuma categoria de vendas detectada com confiança suficiente
  best_score=0.25
  min_confidence=0.3
```

### Logs no Backend (NestJS)

```typescript
// Log quando recebe categoria
[INFO] 💼 Sales category detected: price_interest (confidence: 0.8500)

// Log no processamento
[INFO] 💼 [SALES CATEGORY] Processing sales category: price_interest

// Log na atualização do estado
[INFO] ✅ [TEXT ANALYSIS] Updated with sales category: price_interest (0.85)
```

## Testes

Execute os testes para validar a funcionalidade:

```bash
# Todos os testes
pytest tests/

# Apenas testes de classificação
pytest tests/test_sales_category_classification.py -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

## Referências

- [SBERT Documentation](https://www.sbert.net/)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Hugging Face - paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

