# Guia de Validação - Classificação de Categorias de Vendas

Este guia fornece um checklist completo para validar que a implementação de classificação de categorias de vendas está funcionando corretamente.

## Checklist de Validação

### ✅ Fase 1: Preparação e Estrutura

- [ ] Constante `SALES_CATEGORY_EXAMPLES` definida com 8 categorias
- [ ] Cada categoria tem 10 exemplos em português
- [ ] Atributos `_sales_category_examples_embeddings` e `_sales_examples_loaded` no `__init__`
- [ ] Documentação inline completa nos exemplos

**Validação**:
```python
from src.models.bert_analyzer import SALES_CATEGORY_EXAMPLES

assert len(SALES_CATEGORY_EXAMPLES) == 8
for category, examples in SALES_CATEGORY_EXAMPLES.items():
    assert len(examples) == 10
```

### ✅ Fase 2: Implementação dos Métodos

- [ ] Método `_load_sales_category_examples_embeddings()` implementado
- [ ] Método `classify_sales_category()` implementado
- [ ] Lazy loading funcionando corretamente
- [ ] Cache de embeddings funcionando

**Validação**:
```python
from src.models.bert_analyzer import BERTAnalyzer

analyzer = BERTAnalyzer(
    sbert_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Verificar lazy loading
assert analyzer._sales_examples_loaded == False
categoria, confianca, scores = analyzer.classify_sales_category("Quanto custa?")
assert analyzer._sales_examples_loaded == True
```

### ✅ Fase 3: Integração no Serviço

- [ ] `classify_sales_category()` chamado no `TextAnalysisService.analyze()`
- [ ] Campos `sales_category` e `sales_category_confidence` no resultado
- [ ] Tratamento de erros gracioso (não bloqueia outras análises)
- [ ] Cache funciona corretamente

**Validação**:
```python
from src.services.analysis_service import TextAnalysisService
from src.types.messages import TranscriptionChunk

service = TextAnalysisService()
chunk = TranscriptionChunk(
    meetingId='test',
    participantId='test',
    text='Quanto custa isso?',
    timestamp=1234567890
)

result = await service.analyze(chunk)
assert 'sales_category' in result
assert 'sales_category_confidence' in result
```

### ✅ Fase 4: Logging

- [ ] Logs no Python quando categoria é detectada
- [ ] Logs no backend quando recebe categoria
- [ ] Logs diferenciados (INFO quando presente, DEBUG quando ausente)
- [ ] Contexto completo nos logs

**Validação**:
Verificar logs ao executar análise:
```bash
# Python
grep "sales_category" logs/python.log

# Backend
grep "SALES CATEGORY" logs/backend.log
```

### ✅ Fase 5: Interfaces TypeScript

- [ ] Interface `TextAnalysisResult` atualizada
- [ ] Interface `TextAnalysisEvent` atualizada
- [ ] Tipo `ParticipantState.textAnalysis` atualizado
- [ ] Método `updateStateWithTextAnalysis` atualizado
- [ ] Sem erros de lint TypeScript

**Validação**:
```bash
cd apps/backend
npm run lint
```

### ✅ Fase 6: Testes

- [ ] Testes unitários para `classify_sales_category()`
- [ ] Testes para `_load_sales_category_examples_embeddings()`
- [ ] Testes de integração com `TextAnalysisService`
- [ ] Testes de validação dos exemplos
- [ ] Todos os testes passando

**Validação**:
```bash
cd apps/text-analysis
pytest tests/test_sales_category_classification.py -v
```

## Testes de Validação End-to-End

### Teste 1: Classificação Básica

```bash
# Testar com texto sobre preço
python scripts/validate_sales_category.py "Quanto custa isso?"

# Esperado:
# - Categoria: price_interest
# - Confiança: > 0.5
# - Score de price_interest: maior que outras categorias
```

### Teste 2: Via API REST

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Quanto custa isso?",
    "meetingId": "test_meeting",
    "participantId": "test_participant",
    "timestamp": 1234567890
  }' | jq '.analysis.sales_category'

# Esperado: "price_interest"
```

### Teste 3: Via Socket.IO

```javascript
// No backend NestJS, enviar evento transcription_chunk
socket.emit('transcription_chunk', {
  meetingId: 'test_meeting',
  participantId: 'test_participant',
  text: 'Quanto custa isso?',
  timestamp: Date.now()
});

// Verificar resposta text_analysis_result
socket.on('text_analysis_result', (data) => {
  console.log('Sales category:', data.analysis.sales_category);
  // Esperado: "price_interest"
});
```

### Teste 4: Todas as Categorias

Testar cada categoria com exemplos representativos:

```bash
# price_interest
python scripts/validate_sales_category.py "Quanto custa isso?"

# value_exploration
python scripts/validate_sales_category.py "Como isso vai me ajudar?"

# objection_soft
python scripts/validate_sales_category.py "Preciso pensar melhor"

# objection_hard
python scripts/validate_sales_category.py "Não estou interessado"

# decision_signal
python scripts/validate_sales_category.py "Quando posso começar?"

# information_gathering
python scripts/validate_sales_category.py "Me explique mais sobre isso"

# stalling
python scripts/validate_sales_category.py "Vou pensar sobre isso"

# closing_readiness
python scripts/validate_sales_category.py "Estou pronto para fechar"
```

### Teste 5: Performance

```python
import time
from src.models.bert_analyzer import BERTAnalyzer

analyzer = BERTAnalyzer(
    sbert_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Primeira chamada (deve carregar embeddings)
start = time.time()
categoria, confianca, scores = analyzer.classify_sales_category("Quanto custa?")
first_call_time = time.time() - start
print(f"Primeira chamada: {first_call_time*1000:.2f}ms")

# Chamadas subsequentes (devem usar cache)
times = []
for i in range(10):
    start = time.time()
    categoria, confianca, scores = analyzer.classify_sales_category("Quanto custa?")
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f"Chamadas subsequentes (média): {avg_time*1000:.2f}ms")

# Validar performance
assert first_call_time < 1.0, "Primeira chamada muito lenta"
assert avg_time < 0.1, "Chamadas subsequentes muito lentas"
```

### Teste 6: Casos Extremos

```python
# Texto muito curto
categoria, confianca, scores = analyzer.classify_sales_category("Oi")
# Esperado: categoria pode ser None se score < threshold

# Texto muito longo
texto_longo = " ".join(["Quanto custa?"] * 100)
categoria, confianca, scores = analyzer.classify_sales_category(texto_longo)
# Esperado: deve funcionar (texto será truncado)

# Texto não relacionado
categoria, confianca, scores = analyzer.classify_sales_category("O tempo está bom hoje")
# Esperado: categoria pode ser None ou categoria com baixa confiança

# Sem SBERT configurado
analyzer_no_sbert = BERTAnalyzer(sbert_model_name=None)
categoria, confianca, scores = analyzer_no_sbert.classify_sales_category("Quanto custa?")
# Esperado: categoria = None, confianca = 0.0, scores = {}
```

## Validação de Integração Backend

### Verificar Recebimento no Backend

1. **Iniciar serviços**:
   ```bash
   # Terminal 1: Backend
   cd apps/backend
   npm run start:dev
   
   # Terminal 2: Text Analysis
   cd apps/text-analysis
   docker-compose up
   ```

2. **Enviar transcrição**:
   ```bash
   # Via curl ou usar cliente Socket.IO
   ```

3. **Verificar logs do backend**:
   ```bash
   # Deve aparecer:
   # [INFO] 💼 Sales category detected: price_interest (confidence: 0.8500)
   # [INFO] 💼 [SALES CATEGORY] Processing sales category: price_interest
   # [INFO] ✅ [TEXT ANALYSIS] Updated with sales category: price_interest (0.85)
   ```

4. **Verificar estado do participante**:
   ```typescript
   // No código do backend, verificar:
   const state = feedbackAggregator.getParticipantState(meetingId, participantId);
   console.log('Sales category:', state?.textAnalysis?.sales_category);
   // Esperado: "price_interest"
   ```

## Checklist de Qualidade

### Código

- [ ] Sem erros de lint (Python e TypeScript)
- [ ] Tipos corretos em todas as interfaces
- [ ] Tratamento de erros adequado
- [ ] Logging adequado em todos os pontos críticos

### Performance

- [ ] Primeira classificação: < 1 segundo
- [ ] Classificações subsequentes: < 100ms
- [ ] Memória: < 50MB adicional
- [ ] Cache funcionando corretamente

### Funcionalidade

- [ ] Todas as 8 categorias podem ser detectadas
- [ ] Confiança calculada corretamente (0.0 a 1.0)
- [ ] Threshold mínimo funcionando
- [ ] Comportamento correto sem SBERT configurado

### Integração

- [ ] Dados fluem corretamente Python → Backend
- [ ] Campos aparecem no estado do participante
- [ ] Logs aparecem em ambos os serviços
- [ ] Cache funciona end-to-end

## Problemas Comuns e Soluções

### Problema: `sales_category` sempre `None`

**Causa**: SBERT não configurado ou score abaixo do threshold

**Solução**:
1. Verificar `SBERT_MODEL_NAME` está definido
2. Reduzir `min_confidence` para 0.3 ou menos
3. Verificar logs para ver scores de todas as categorias

### Problema: Classificação incorreta

**Causa**: Texto ambíguo ou exemplos não cobrem variação

**Solução**:
1. Verificar confiança (deve ser > 0.5)
2. Adicionar mais exemplos à categoria relevante
3. Ajustar threshold se necessário

### Problema: Performance lenta

**Causa**: Primeira chamada ou modelo em CPU

**Solução**:
1. Primeira chamada é esperada (~400ms)
2. Configurar GPU se disponível
3. Verificar que cache está funcionando

## Relatório de Validação

Após executar todos os testes, preencha:

```
✅ Fase 1: Preparação - [PASS/FAIL]
✅ Fase 2: Implementação - [PASS/FAIL]
✅ Fase 3: Integração - [PASS/FAIL]
✅ Fase 4: Logging - [PASS/FAIL]
✅ Fase 5: Interfaces TypeScript - [PASS/FAIL]
✅ Fase 6: Testes - [PASS/FAIL]

Testes End-to-End:
- Teste 1: Classificação Básica - [PASS/FAIL]
- Teste 2: Via API REST - [PASS/FAIL]
- Teste 3: Via Socket.IO - [PASS/FAIL]
- Teste 4: Todas as Categorias - [PASS/FAIL]
- Teste 5: Performance - [PASS/FAIL]
- Teste 6: Casos Extremos - [PASS/FAIL]

Observações:
[Anotar problemas encontrados ou melhorias sugeridas]
```

