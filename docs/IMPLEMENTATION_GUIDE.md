# Guia de Implementação - Roadmap de Melhorias

**Versão**: 1.0  
**Data**: 2025-01-XX  
**Baseado em**: `IMPROVEMENT_ROADMAP.md`

---

## 📋 Visão Geral

Este documento fornece um passo a passo detalhado para implementar as melhorias descritas no roadmap. As tarefas estão organizadas por fases, com dependências claras e critérios de aceitação.

**Estrutura de Implementação**:
- **Fase 1**: Quick Wins (Semana 1-2)
- **Fase 2**: Melhorias Semânticas Básicas (Semana 3-4)
- **Fase 3**: Contexto Conversacional (Semana 5-7)
- **Fase 4**: Integração Backend (Semana 8-9)
- **Fase 5**: Observabilidade (Semana 10)

---

## 🎯 Fase 1: Quick Wins (Semana 1-2)

### Tarefa 1.1: Expandir Exemplos de Referência

**Objetivo**: Adicionar mais exemplos por categoria para melhorar precisão

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Localizar `SALES_CATEGORY_EXAMPLES`** (linha ~51)
   ```python
   SALES_CATEGORY_EXAMPLES: Dict[str, List[str]] = {
       'price_interest': [...],
       ...
   }
   ```

2. **Adicionar 5 exemplos por categoria** (total: 80 → 120)
   - Incluir variações regionais (Brasil vs Portugal)
   - Incluir gírias e expressões informais
   - Incluir formas mais indiretas de expressar a categoria

3. **Validar exemplos**:
   - Garantir que exemplos são realmente da categoria
   - Evitar exemplos ambíguos
   - Testar com script de validação

**Critérios de Aceitação**:
- ✅ 15 exemplos por categoria (120 total)
- ✅ Exemplos cobrem variações linguísticas
- ✅ Testes unitários passam
- ✅ Script de validação funciona

**Estimativa**: 4 horas

**Testes**:
```bash
# Executar testes existentes
cd apps/text-analysis
python -m pytest tests/test_sales_category_classification.py -v

# Validar manualmente
python scripts/validate_sales_category.py
```

---

### Tarefa 1.2: Adicionar Score de Ambiguidade

**Objetivo**: Calcular métrica de ambiguidade semântica

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Adicionar método `_calculate_ambiguity`** após `classify_sales_category`:
   ```python
   def _calculate_ambiguity(self, scores: Dict[str, float]) -> float:
       """
       Calcula ambiguidade baseada na distribuição dos scores.
       
       Returns:
           float: 0.0 (claro) a 1.0 (muito ambíguo)
       """
       if not scores:
           return 1.0
       
       sorted_scores = sorted(scores.values(), reverse=True)
       
       if len(sorted_scores) < 2:
           return 0.0
       
       # Entropia normalizada dos scores
       import numpy as np
       scores_array = np.array(sorted_scores)
       scores_normalized = scores_array / scores_array.sum()
       entropy = -np.sum(scores_normalized * np.log(scores_normalized + 1e-10))
       max_entropy = np.log(len(scores))
       
       return entropy / max_entropy if max_entropy > 0 else 0.0
   ```

2. **Modificar `classify_sales_category`** para retornar ambiguidade:
   ```python
   def classify_sales_category(...) -> Tuple[Optional[str], float, Dict[str, float], float]:
       # ... código existente ...
       
       # Calcular ambiguidade
       ambiguity = self._calculate_ambiguity(scores)
       
       return categoria, confianca, scores, ambiguity
   ```

3. **Atualizar chamadas** em `analysis_service.py`:
   ```python
   categoria, confianca, scores, ambiguity = analyzer.classify_sales_category(...)
   ```

4. **Incluir ambiguidade no resultado**:
   ```python
   result = {
       # ... campos existentes ...
       'sales_category_ambiguity': ambiguity
   }
   ```

**Critérios de Aceitação**:
- ✅ Método `_calculate_ambiguity` implementado
- ✅ Ambiguidade retornada em `classify_sales_category`
- ✅ Ambiguidade incluída no resultado final
- ✅ Testes unitários passam
- ✅ Ambiguidade varia de 0.0 a 1.0

**Estimativa**: 2 horas

**Testes**:
```python
# Teste unitário
def test_ambiguity_calculation():
    analyzer = BERTAnalyzer()
    
    # Caso claro (uma categoria dominante)
    scores_clear = {'price_interest': 0.9, 'value_exploration': 0.1}
    ambiguity = analyzer._calculate_ambiguity(scores_clear)
    assert ambiguity < 0.3
    
    # Caso ambíguo (scores próximos)
    scores_ambiguous = {'price_interest': 0.5, 'value_exploration': 0.48}
    ambiguity = analyzer._calculate_ambiguity(scores_ambiguous)
    assert ambiguity > 0.7
```

---

### Tarefa 1.3: Adicionar Intensidade do Sinal

**Objetivo**: Adicionar score absoluto da melhor categoria (diferente de confiança)

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`
- `apps/text-analysis/src/services/analysis_service.py`

**Passos**:

1. **Modificar `classify_sales_category`** para retornar intensidade:
   ```python
   def classify_sales_category(...) -> Tuple[Optional[str], float, Dict[str, float], float, float]:
       # ... código existente ...
       
       # Intensidade = score absoluto da melhor categoria
       intensity = max(scores.values()) if scores else 0.0
       
       return categoria, confianca, scores, ambiguity, intensity
   ```

2. **Atualizar `analysis_service.py`**:
   ```python
   categoria, confianca, scores, ambiguity, intensity = analyzer.classify_sales_category(...)
   
   result = {
       # ... campos existentes ...
       'sales_category_intensity': intensity
   }
   ```

**Critérios de Aceitação**:
- ✅ Intensidade retornada em `classify_sales_category`
- ✅ Intensidade incluída no resultado final
- ✅ Intensidade = score absoluto da melhor categoria
- ✅ Testes unitários passam

**Estimativa**: 1 hora

---

### Tarefa 1.4: Melhorar Logging Explicável

**Objetivo**: Logs mais detalhados explicando decisões semânticas

**Arquivos a Modificar**:
- `apps/text-analysis/src/services/analysis_service.py`

**Passos**:

1. **Melhorar log após classificação** (linha ~328):
   ```python
   if sales_category:
       logger.info(
           "✅ [ANÁLISE] Categoria de vendas classificada",
           meeting_id=chunk.meetingId,
           participant_id=chunk.participantId,
           sales_category=sales_category,
           sales_category_confidence=round(confianca, 4),
           sales_category_intensity=round(intensity, 4),
           sales_category_ambiguity=round(ambiguity, 4),
           best_score=round(scores.get(sales_category, 0.0), 4) if scores else 0.0,
           reasoning={
               "why": f"High confidence {sales_category} classification",
               "confidence_reason": "Large gap between best and second best",
               "intensity_reason": f"Absolute score of {intensity:.2f}",
               "ambiguity_reason": f"Low ambiguity ({ambiguity:.2f})"
           }
       )
   ```

2. **Adicionar log quando categoria não detectada**:
   ```python
   else:
       logger.debug(
           "⚠️ [ANÁLISE] Nenhuma categoria detectada",
           meeting_id=chunk.meetingId,
           best_score=round(max(scores.values()) if scores else 0.0, 4),
           ambiguity=round(ambiguity, 4),
           reasoning={
               "why": "No category met minimum confidence threshold",
               "reason": f"Best score {max(scores.values()):.2f} < {min_confidence}"
           }
       )
   ```

**Critérios de Aceitação**:
- ✅ Logs incluem reasoning detalhado
- ✅ Logs explicam por que categoria foi/não foi detectada
- ✅ Logs incluem todas as métricas (confiança, intensidade, ambiguidade)

**Estimativa**: 2 horas

---

### Tarefa 1.5: Flags Semânticas Básicas

**Objetivo**: Implementar flags booleanas para facilitar heurísticas no backend

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`
- `apps/text-analysis/src/services/analysis_service.py`

**Passos**:

1. **Adicionar método `_generate_semantic_flags`** em `BERTAnalyzer`:
   ```python
   def _generate_semantic_flags(
       self,
       category: Optional[str],
       confidence: float,
       intensity: float,
       ambiguity: float
   ) -> Dict[str, bool]:
       """
       Gera flags semânticas baseadas em análise.
       
       Returns:
           Dict com flags booleanas
       """
       flags = {}
       
       if not category:
           return flags
       
       # Flag: Janela de oportunidade para preço
       flags['price_window_open'] = (
           category == 'price_interest' and
           confidence > 0.7 and
           intensity > 0.8 and
           ambiguity < 0.3
       )
       
       # Flag: Sinal forte de decisão
       flags['decision_signal_strong'] = (
           category in ['decision_signal', 'closing_readiness'] and
           confidence > 0.8 and
           intensity > 0.85 and
           ambiguity < 0.25
       )
       
       # Flag: Pronto para fechar
       flags['ready_to_close'] = (
           category == 'closing_readiness' and
           confidence > 0.85 and
           intensity > 0.9 and
           ambiguity < 0.2
       )
       
       return flags
   ```

2. **Integrar flags em `classify_sales_category`**:
   ```python
   def classify_sales_category(...):
       # ... código existente ...
       
       flags = self._generate_semantic_flags(categoria, confianca, intensity, ambiguity)
       
       return categoria, confianca, scores, ambiguity, intensity, flags
   ```

3. **Incluir flags no resultado** em `analysis_service.py`:
   ```python
   categoria, confianca, scores, ambiguity, intensity, flags = analyzer.classify_sales_category(...)
   
   result = {
       # ... campos existentes ...
       'sales_category_flags': flags
   }
   ```

**Critérios de Aceitação**:
- ✅ Método `_generate_semantic_flags` implementado
- ✅ 3 flags básicas implementadas
- ✅ Flags incluídas no resultado final
- ✅ Testes unitários passam

**Estimativa**: 4 horas

**Testes**:
```python
def test_semantic_flags():
    analyzer = BERTAnalyzer()
    
    # Teste price_window_open
    flags = analyzer._generate_semantic_flags(
        'price_interest', 0.85, 0.9, 0.2
    )
    assert flags['price_window_open'] == True
    
    # Teste decision_signal_strong
    flags = analyzer._generate_semantic_flags(
        'decision_signal', 0.9, 0.88, 0.2
    )
    assert flags['decision_signal_strong'] == True
```

---

### Checklist Fase 1

- [ ] Tarefa 1.1: Exemplos expandidos
- [ ] Tarefa 1.2: Ambiguidade implementada
- [ ] Tarefa 1.3: Intensidade implementada
- [ ] Tarefa 1.4: Logging melhorado
- [ ] Tarefa 1.5: Flags básicas implementadas
- [ ] Todos os testes passam
- [ ] Validação manual realizada
- [ ] Documentação atualizada

**Total Fase 1**: ~13 horas

---

## 🔄 Fase 2: Melhorias Semânticas Básicas (Semana 3-4)

### Tarefa 2.1: Atualizar Interfaces TypeScript (Backend)

**Objetivo**: Atualizar interfaces para receber novos campos

**Arquivos a Modificar**:
- `apps/backend/src/pipeline/text-analysis.service.ts`
- `apps/backend/src/feedback/feedback.types.ts`
- `apps/backend/src/feedback/a2e2/types.ts`
- `apps/backend/src/feedback/feedback.aggregator.service.ts`

**Passos**:

1. **Atualizar `TextAnalysisResult`** em `text-analysis.service.ts`:
   ```typescript
   export interface TextAnalysisResult {
     // ... campos existentes ...
     analysis: {
       // ... campos existentes ...
       sales_category?: string | null;
       sales_category_confidence?: number | null;
       sales_category_intensity?: number | null;      // NOVO
       sales_category_ambiguity?: number | null;      // NOVO
       sales_category_flags?: {                       // NOVO
         price_window_open?: boolean;
         decision_signal_strong?: boolean;
         ready_to_close?: boolean;
       } | null;
     };
   }
   ```

2. **Atualizar `TextAnalysisEvent`** em `feedback.types.ts`:
   ```typescript
   export interface TextAnalysisEvent {
     // ... campos existentes ...
     analysis: {
       // ... campos existentes ...
       sales_category_intensity?: number | null;
       sales_category_ambiguity?: number | null;
       sales_category_flags?: {
         price_window_open?: boolean;
         decision_signal_strong?: boolean;
         ready_to_close?: boolean;
       } | null;
     };
   }
   ```

3. **Atualizar `ParticipantState`** em `a2e2/types.ts`:
   ```typescript
   textAnalysis: {
     // ... campos existentes ...
     sales_category_intensity?: number | null;
     sales_category_ambiguity?: number | null;
     sales_category_flags?: {
       price_window_open?: boolean;
       decision_signal_strong?: boolean;
       ready_to_close?: boolean;
     } | null;
   }
   ```

4. **Atualizar `feedback.aggregator.service.ts`**:
   ```typescript
   state.textAnalysis = {
     // ... campos existentes ...
     sales_category_intensity: evt.analysis.sales_category_intensity ?? undefined,
     sales_category_ambiguity: evt.analysis.sales_category_ambiguity ?? undefined,
     sales_category_flags: evt.analysis.sales_category_flags ?? undefined,
   };
   ```

**Critérios de Aceitação**:
- ✅ Todas as interfaces atualizadas
- ✅ Campos opcionais (não quebram código existente)
- ✅ TypeScript compila sem erros
- ✅ Logs incluem novos campos

**Estimativa**: 2 horas

---

### Tarefa 2.2: Classificação Multi-Label (Opcional)

**Objetivo**: Permitir múltiplas categorias quando apropriado

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Adicionar método `classify_sales_category_multi`**:
   ```python
   def classify_sales_category_multi(
       self,
       text: str,
       min_confidence: float = 0.3,
       max_categories: int = 2,
       similarity_threshold: float = 0.7
   ) -> Tuple[List[Tuple[str, float]], float, Dict[str, float]]:
       """
       Classifica texto em múltiplas categorias quando scores são próximos.
       
       Args:
           text: Texto a classificar
           min_confidence: Score mínimo para aceitar categoria
           max_categories: Máximo de categorias a retornar
           similarity_threshold: Se segunda melhor tem score > threshold × melhor, incluir ambas
       
       Returns:
           Lista de (categoria, score) ordenada por score
           Confiança geral
           Scores de todas as categorias
       """
       # Usar método existente para obter scores
       categoria, confianca, scores, ambiguity, intensity, flags = \
           self.classify_sales_category(text, min_confidence)
       
       if not scores:
           return [], 0.0, {}
       
       # Ordenar por score
       sorted_categories = sorted(
           scores.items(),
           key=lambda x: x[1],
           reverse=True
       )
       
       # Determinar quantas categorias incluir
       if len(sorted_categories) < 2:
           return sorted_categories, confianca, scores
       
       best_score = sorted_categories[0][1]
       second_score = sorted_categories[1][1]
       
       # Se segunda melhor está próxima da melhor, incluir ambas
       if second_score >= similarity_threshold * best_score and \
          second_score >= min_confidence:
           return sorted_categories[:max_categories], confianca, scores
       
       # Caso contrário, apenas a melhor
       return [sorted_categories[0]], confianca, scores
   ```

2. **Adicionar flag para usar multi-label** em `analysis_service.py`:
   ```python
   # Opcional: usar multi-label
   use_multi_label = False  # Configurar via env se necessário
   
   if use_multi_label:
       categories, confianca, scores = analyzer.classify_sales_category_multi(
           chunk.text,
           min_confidence=0.3
       )
       sales_category = categories[0][0] if categories else None
       sales_category_secondary = [cat for cat, _ in categories[1:]]
   else:
       # Usar método atual
       categoria, confianca, scores, ambiguity, intensity, flags = \
           analyzer.classify_sales_category(chunk.text, min_confidence=0.3)
   ```

**Critérios de Aceitação**:
- ✅ Método `classify_sales_category_multi` implementado
- ✅ Retorna múltiplas categorias quando apropriado
- ✅ Testes unitários passam
- ✅ Opcional (não quebra código existente)

**Estimativa**: 8 horas

**Testes**:
```python
def test_multi_label():
    analyzer = BERTAnalyzer()
    
    # Texto que pode ser múltiplas categorias
    text = "Quanto custa e como funciona?"
    categories, conf, scores = analyzer.classify_sales_category_multi(text)
    
    assert len(categories) >= 1
    assert 'price_interest' in [cat for cat, _ in categories]
```

---

### Checklist Fase 2

- [ ] Tarefa 2.1: Interfaces TypeScript atualizadas
- [ ] Tarefa 2.2: Multi-label implementado (opcional)
- [ ] Backend recebe novos campos corretamente
- [ ] Testes de integração passam
- [ ] Validação end-to-end realizada

**Total Fase 2**: ~10 horas

---

## 🗣️ Fase 3: Contexto Conversacional (Semana 5-7)

### Tarefa 3.1: Implementar ConversationContext

**Objetivo**: Manter histórico semântico da conversa

**Arquivos a Criar**:
- `apps/text-analysis/src/models/conversation_context.py`

**Arquivos a Modificar**:
- `apps/text-analysis/src/services/analysis_service.py`

**Passos**:

1. **Criar `conversation_context.py`**:
   ```python
   from typing import Dict, List, Optional, Any
   import time
   
   class ConversationContext:
       """
       Mantém contexto semântico da conversa.
       """
       def __init__(self, window_size: int = 10, window_duration_ms: int = 60000):
           self.window_size = window_size  # Últimos N chunks
           self.window_duration_ms = window_duration_ms  # Últimos N segundos
           self.history: List[Dict[str, Any]] = []
       
       def add_chunk(self, chunk: Dict[str, Any]):
           """Adiciona chunk ao histórico"""
           self.history.append({
               'text': chunk['text'],
               'sales_category': chunk.get('sales_category'),
               'sales_category_confidence': chunk.get('sales_category_confidence'),
               'sales_category_intensity': chunk.get('sales_category_intensity'),
               'sales_category_ambiguity': chunk.get('sales_category_ambiguity'),
               'timestamp': chunk['timestamp'],
               'embedding': chunk.get('embedding')
           })
           self._prune_history()
       
       def _prune_history(self):
           """Remove chunks antigos da janela"""
           if not self.history:
               return
           
           # Remover por tamanho
           if len(self.history) > self.window_size:
               self.history = self.history[-self.window_size:]
           
           # Remover por tempo (se necessário)
           if self.history:
               now = self.history[-1]['timestamp']
               cutoff = now - self.window_duration_ms
               self.history = [
                   chunk for chunk in self.history
                   if chunk['timestamp'] >= cutoff
               ]
       
       def get_window(self, now: Optional[int] = None) -> List[Dict[str, Any]]:
           """Retorna chunks na janela temporal"""
           if not self.history:
               return []
           
           if now is None:
               now = self.history[-1]['timestamp'] if self.history else 0
           
           cutoff = now - self.window_duration_ms
           return [
               chunk for chunk in self.history
               if chunk['timestamp'] >= cutoff
           ][-self.window_size:]
       
       def clear(self):
           """Limpa histórico (útil para testes)"""
           self.history = []
   ```

2. **Integrar em `TextAnalysisService`**:
   ```python
   class TextAnalysisService:
       def __init__(self):
           # ... código existente ...
           self.conversation_contexts: Dict[str, ConversationContext] = {}
       
       def _get_context_key(self, chunk: TranscriptionChunk) -> str:
           """Gera chave única para contexto"""
           return f"{chunk.meetingId}:{chunk.participantId}"
       
       async def analyze(self, chunk: TranscriptionChunk):
           # ... análise atual ...
           
           # Adicionar ao contexto
           context_key = self._get_context_key(chunk)
           if context_key not in self.conversation_contexts:
               self.conversation_contexts[context_key] = ConversationContext()
           
           context = self.conversation_contexts[context_key]
           context.add_chunk({
               'text': chunk.text,
               'sales_category': sales_category,
               'sales_category_confidence': sales_category_confidence,
               'sales_category_intensity': sales_category_intensity,
               'sales_category_ambiguity': sales_category_ambiguity,
               'timestamp': chunk.timestamp,
               'embedding': embedding
           })
           
           # Obter janela para análise contextual (será usado nas próximas tarefas)
           window = context.get_window(chunk.timestamp)
           
           # ... resto do código ...
   ```

**Critérios de Aceitação**:
- ✅ Classe `ConversationContext` implementada
- ✅ Histórico mantido por participante/reunião
- ✅ Janela temporal funciona corretamente
- ✅ Testes unitários passam

**Estimativa**: 4 horas

**Testes**:
```python
def test_conversation_context():
    ctx = ConversationContext(window_size=5, window_duration_ms=60000)
    
    # Adicionar chunks
    for i in range(10):
        ctx.add_chunk({
            'text': f'chunk {i}',
            'sales_category': 'price_interest',
            'timestamp': i * 1000
        })
    
    # Deve manter apenas últimos 5
    assert len(ctx.history) == 5
    assert ctx.history[0]['text'] == 'chunk 5'
```

---

### Tarefa 3.2: Agregação Temporal de Categorias

**Objetivo**: Agregar categorias em janela temporal para reduzir ruído

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Adicionar método `aggregate_categories_temporal`**:
   ```python
   def aggregate_categories_temporal(
       self,
       window: List[Dict[str, Any]]
   ) -> Optional[Dict[str, Any]]:
       """
       Agrega categorias em janela temporal.
       
       Returns:
           {
               'dominant_category': str,
               'category_distribution': Dict[str, float],
               'stability': float,
               'trend': 'advancing' | 'stable' | 'regressing'
           }
       """
       if not window:
           return None
       
       # Contar ocorrências de cada categoria
       category_counts = {}
       for chunk in window:
           cat = chunk.get('sales_category')
           if cat:
               category_counts[cat] = category_counts.get(cat, 0) + 1
       
       if not category_counts:
           return None
       
       # Calcular distribuição
       total = sum(category_counts.values())
       distribution = {
           cat: count / total
           for cat, count in category_counts.items()
       }
       
       # Categoria dominante
       dominant = max(category_counts.items(), key=lambda x: x[1])[0]
       
       # Estabilidade (quanto mais concentrada, mais estável)
       max_prob = max(distribution.values())
       stability = max_prob
       
       return {
           'dominant_category': dominant,
           'category_distribution': distribution,
           'stability': stability
       }
   ```

2. **Integrar em `analysis_service.py`**:
   ```python
   # Após adicionar ao contexto
   window = context.get_window(chunk.timestamp)
   
   # Agregar categorias temporais
   aggregated = analyzer.aggregate_categories_temporal(window)
   
   # Incluir no resultado
   result = {
       # ... campos existentes ...
       'sales_category_aggregated': aggregated
   }
   ```

**Critérios de Aceitação**:
- ✅ Método `aggregate_categories_temporal` implementado
- ✅ Calcula categoria dominante corretamente
- ✅ Calcula estabilidade corretamente
- ✅ Testes unitários passam

**Estimativa**: 4 horas

---

### Tarefa 3.3: Detecção de Transições

**Objetivo**: Detectar mudanças significativas de categoria

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Adicionar constante de progressão**:
   ```python
   CATEGORY_PROGRESSION = {
       'information_gathering': 1,
       'value_exploration': 2,
       'price_interest': 3,
       'decision_signal': 4,
       'closing_readiness': 5,
       'stalling': 0,
       'objection_soft': -1,
       'objection_hard': -2
   }
   ```

2. **Adicionar método `detect_category_transition`**:
   ```python
   def detect_category_transition(
       self,
       current_category: Optional[str],
       current_score: float,
       history: List[Dict[str, Any]]
   ) -> Optional[Dict[str, Any]]:
       """
       Detecta transições significativas de categoria.
       
       Returns:
           {
               'transition_type': 'advancing' | 'regressing' | 'lateral',
               'from_category': str,
               'to_category': str,
               'confidence': float,
               'time_delta_ms': int
           } ou None
       """
       if not current_category or not history:
           return None
       
       # Obter categoria anterior (última com categoria válida)
       previous_category = None
       previous_timestamp = None
       for chunk in reversed(history):
           cat = chunk.get('sales_category')
           if cat:
               previous_category = cat
               previous_timestamp = chunk.get('timestamp')
               break
       
       if not previous_category or previous_category == current_category:
           return None
       
       # Determinar tipo de transição
       current_stage = CATEGORY_PROGRESSION.get(current_category, 0)
       previous_stage = CATEGORY_PROGRESSION.get(previous_category, 0)
       
       if current_stage > previous_stage:
           transition_type = 'advancing'
       elif current_stage < previous_stage:
           transition_type = 'regressing'
       else:
           transition_type = 'lateral'
       
       # Calcular confiança da transição
       # Baseado na diferença de estágios e scores
       stage_diff = abs(current_stage - previous_stage)
       confidence = min(1.0, stage_diff / 3.0) * current_score
       
       # Calcular tempo decorrido
       current_timestamp = history[-1].get('timestamp') if history else 0
       time_delta_ms = current_timestamp - previous_timestamp if previous_timestamp else 0
       
       return {
           'transition_type': transition_type,
           'from_category': previous_category,
           'to_category': current_category,
           'confidence': confidence,
           'time_delta_ms': time_delta_ms
       }
   ```

3. **Integrar em `analysis_service.py`**:
   ```python
   # Detectar transição
   transition = analyzer.detect_category_transition(
       sales_category,
       sales_category_confidence or 0.0,
       window
   )
   
   result = {
       # ... campos existentes ...
       'sales_category_transition': transition
   }
   ```

**Critérios de Aceitação**:
- ✅ Método `detect_category_transition` implementado
- ✅ Detecta transições corretamente
- ✅ Classifica tipo de transição (advancing/regressing/lateral)
- ✅ Testes unitários passam

**Estimativa**: 6 horas

---

### Tarefa 3.4: Tendência Semântica

**Objetivo**: Calcular tendência da conversa ao longo do tempo

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Adicionar método `calculate_semantic_trend`**:
   ```python
   def calculate_semantic_trend(
       self,
       window: List[Dict[str, Any]]
   ) -> Dict[str, Any]:
       """
       Calcula tendência semântica da conversa.
       
       Returns:
           {
               'trend': 'advancing' | 'stable' | 'regressing',
               'trend_strength': float,
               'current_stage': int,
               'velocity': float
           }
       """
       if len(window) < 2:
           return {
               'trend': 'stable',
               'trend_strength': 0.0,
               'current_stage': 0,
               'velocity': 0.0
           }
       
       # Mapear categorias para números
       progression_values = []
       for chunk in window:
           cat = chunk.get('sales_category')
           if cat:
               stage = CATEGORY_PROGRESSION.get(cat, 0)
               progression_values.append(stage)
       
       if len(progression_values) < 2:
           return {
               'trend': 'stable',
               'trend_strength': 0.0,
               'current_stage': progression_values[-1] if progression_values else 0,
               'velocity': 0.0
           }
       
       # Calcular tendência (regressão linear simples)
       import numpy as np
       x = np.arange(len(progression_values))
       y = np.array(progression_values)
       
       slope = np.polyfit(x, y, 1)[0]
       
       # Normalizar slope para [-1, 1]
       trend_strength = min(1.0, abs(slope) / 2.0)
       
       if slope > 0.1:
           trend = 'advancing'
       elif slope < -0.1:
           trend = 'regressing'
       else:
           trend = 'stable'
       
       return {
           'trend': trend,
           'trend_strength': trend_strength,
           'current_stage': progression_values[-1] if progression_values else 0,
           'velocity': slope
       }
   ```

2. **Integrar em `analysis_service.py`**:
   ```python
   # Calcular tendência
   trend = analyzer.calculate_semantic_trend(window)
   
   result = {
       # ... campos existentes ...
       'sales_category_trend': trend
   }
   ```

**Critérios de Aceitação**:
- ✅ Método `calculate_semantic_trend` implementado
- ✅ Calcula tendência corretamente
- ✅ Retorna direção (advancing/stable/regressing)
- ✅ Testes unitários passam

**Estimativa**: 6 horas

---

### Tarefa 3.5: Redução de Ruído com Consistência

**Objetivo**: Usar contexto para reduzir falsos positivos

**Arquivos a Modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py`

**Passos**:

1. **Adicionar método `classify_with_context`**:
   ```python
   def classify_with_context(
       self,
       text: str,
       context_window: List[Dict[str, Any]],
       min_consistency: float = 0.6,
       min_confidence: float = 0.3
   ) -> Dict[str, Any]:
       """
       Classifica texto considerando contexto histórico.
       
       Returns:
           {
               'category': str,
               'confidence': float,
               'is_consistent': bool,
               'used_context': bool
           }
       """
       # Classificar chunk atual
       categoria, confianca, scores, ambiguity, intensity, flags = \
           self.classify_sales_category(text, min_confidence)
       
       if not context_window:
           return {
               'category': categoria,
               'confidence': confianca,
               'is_consistent': True,
               'used_context': False
           }
       
       # Agregar categorias do histórico
       aggregated = self.aggregate_categories_temporal(context_window)
       
       if not aggregated:
           return {
               'category': categoria,
               'confidence': confianca,
               'is_consistent': True,
               'used_context': False
           }
       
       dominant_historical = aggregated['dominant_category']
       stability = aggregated['stability']
       
       # Verificar consistência
       is_consistent = (
           categoria == dominant_historical or
           stability < 0.5  # Histórico instável, aceitar atual
       )
       
       if is_consistent or confianca > 0.8:
           # Usar categoria atual
           return {
               'category': categoria,
               'confidence': confianca,
               'is_consistent': is_consistent,
               'used_context': False
           }
       else:
           # Usar categoria histórica (mais confiável)
           return {
               'category': dominant_historical,
               'confidence': stability,
               'is_consistent': False,
               'used_context': True
           }
   ```

2. **Integrar em `analysis_service.py`** (opcional, pode ser flag):
   ```python
   # Opção: usar classificação com contexto
   use_contextual_classification = True  # Configurar via env
   
   if use_contextual_classification and window:
       contextual_result = analyzer.classify_with_context(
           chunk.text,
           window,
           min_consistency=0.6
       )
       
       # Usar categoria contextual se mais confiável
       if contextual_result['used_context']:
           sales_category = contextual_result['category']
           sales_category_confidence = contextual_result['confidence']
   ```

**Critérios de Aceitação**:
- ✅ Método `classify_with_context` implementado
- ✅ Usa contexto quando apropriado
- ✅ Reduz falsos positivos
- ✅ Testes unitários passam

**Estimativa**: 6 horas

---

### Checklist Fase 3

- [ ] Tarefa 3.1: ConversationContext implementado
- [ ] Tarefa 3.2: Agregação temporal implementada
- [ ] Tarefa 3.3: Detecção de transições implementada
- [ ] Tarefa 3.4: Tendência semântica implementada
- [ ] Tarefa 3.5: Redução de ruído implementada
- [ ] Todos os testes passam
- [ ] Performance aceitável (< 100ms adicional)

**Total Fase 3**: ~26 horas

---

## 🔌 Fase 4: Integração Backend (Semana 8-9)

### Tarefa 4.1: Atualizar Interfaces para Contexto

**Objetivo**: Backend recebe dados de contexto

**Arquivos a Modificar**:
- `apps/backend/src/pipeline/text-analysis.service.ts`
- `apps/backend/src/feedback/feedback.types.ts`
- `apps/backend/src/feedback/a2e2/types.ts`

**Passos**:

1. **Atualizar interfaces** para incluir:
   - `sales_category_aggregated`
   - `sales_category_transition`
   - `sales_category_trend`

2. **Atualizar `ParticipantState`** para armazenar histórico

**Critérios de Aceitação**:
- ✅ Interfaces atualizadas
- ✅ TypeScript compila
- ✅ Dados recebidos corretamente

**Estimativa**: 2 horas

---

### Tarefa 4.2: Implementar Heurísticas de Feedback

**Objetivo**: Backend usa sinais semânticos para gerar feedbacks

**Arquivos a Modificar**:
- `apps/backend/src/feedback/feedback.aggregator.service.ts`

**Passos**:

1. **Implementar `shouldGenerateSalesFeedback`**:
   ```typescript
   function shouldGenerateSalesFeedback(
     state: ParticipantState,
     semanticSignals: SemanticSignals,
     now: number
   ): boolean {
     // Verificar cooldown
     // Verificar flags fortes
     // Verificar consistência
     // Verificar confiança e intensidade
     // Verificar ambiguidade
   }
   ```

2. **Implementar `generateSalesFeedback`**:
   ```typescript
   function generateSalesFeedback(
     state: ParticipantState,
     semanticSignals: SemanticSignals
   ): FeedbackEventPayload | null {
     // Heurísticas baseadas em flags
     // Heurísticas baseadas em transições
     // Heurísticas baseadas em tendência
   }
   ```

3. **Implementar priorização**:
   ```typescript
   function prioritizeFeedback(feedback: FeedbackEventPayload): number {
     // Retornar prioridade numérica
   }
   ```

**Critérios de Aceitação**:
- ✅ Heurísticas implementadas
- ✅ Feedbacks gerados corretamente
- ✅ Priorização funciona
- ✅ Testes passam

**Estimativa**: 12 horas

---

### Checklist Fase 4

- [ ] Tarefa 4.1: Interfaces atualizadas
- [ ] Tarefa 4.2: Heurísticas implementadas
- [ ] Feedbacks gerados corretamente
- [ ] Testes de integração passam

**Total Fase 4**: ~14 horas

---

## 📊 Fase 5: Observabilidade (Semana 10)

### Tarefa 5.1: Métricas Semânticas

**Objetivo**: Coletar métricas de qualidade

**Arquivos a Criar**:
- `apps/text-analysis/src/metrics/semantic_metrics.py`

**Arquivos a Modificar**:
- `apps/text-analysis/src/services/analysis_service.py`
- `apps/text-analysis/src/main.py`

**Passos**:

1. **Criar `SemanticMetrics`** (ver roadmap para implementação)

2. **Integrar em `TextAnalysisService`**

3. **Expor endpoint `/metrics`** no FastAPI

**Critérios de Aceitação**:
- ✅ Métricas coletadas
- ✅ Endpoint `/metrics` funciona
- ✅ Métricas úteis para análise

**Estimativa**: 6 horas

---

### Checklist Fase 5

- [ ] Tarefa 5.1: Métricas implementadas
- [ ] Endpoint `/metrics` funciona
- [ ] Logs melhorados

**Total Fase 5**: ~6 horas

---

## 📝 Resumo de Implementação

### Ordem de Execução Recomendada

1. **Semana 1-2**: Fase 1 (Quick Wins)
2. **Semana 3-4**: Fase 2 (Melhorias Semânticas)
3. **Semana 5-7**: Fase 3 (Contexto Conversacional)
4. **Semana 8-9**: Fase 4 (Integração Backend)
5. **Semana 10**: Fase 5 (Observabilidade)

### Total Estimado

- **Fase 1**: 13 horas
- **Fase 2**: 10 horas
- **Fase 3**: 26 horas
- **Fase 4**: 14 horas
- **Fase 5**: 6 horas
- **Total**: ~69 horas (~2 semanas de trabalho full-time)

### Dependências Críticas

- Fase 2 depende de Fase 1 (interfaces precisam dos novos campos)
- Fase 3 depende de Fase 1 (usa intensidade, ambiguidade, flags)
- Fase 4 depende de Fase 3 (usa contexto e transições)
- Fase 5 pode ser feita em paralelo

### Validação Contínua

Após cada fase:
1. Executar testes unitários
2. Executar testes de integração
3. Validação manual com script
4. Verificar performance (latência)
5. Verificar logs

### Próximos Passos Após Implementação

1. Coletar dados reais de uso
2. Ajustar thresholds baseado em dados
3. Melhorar exemplos baseado em falsos positivos
4. Considerar fine-tuning do modelo (longo prazo)

---

**Documento criado em**: 2025-01-XX  
**Versão**: 1.0  
**Status**: Guia de Implementação

