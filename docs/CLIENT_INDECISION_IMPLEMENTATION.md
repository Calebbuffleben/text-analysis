# Planejamento: Detecção de Indecisão do Cliente

**Objetivo**: Implementar feedback para detectar padrão consistente de indecisão do cliente, caracterizado por postergar decisões, solicitar mais tempo, repetir dúvidas e evitar compromissos claros.

**Status**: 📋 Planejamento  
**Data**: 2025-01-XX

---

## 📊 Visão Geral

### O que será implementado

**Backend (TypeScript) - Obrigatório**:
1. **Armazenamento de histórico de textos** no `ParticipantState`
2. **Detecção de padrões semânticos** de indecisão
3. **Extração de frases representativas** do histórico
4. **Cálculo de consistência temporal** do padrão
5. **Cálculo de confidence** combinando múltiplos sinais
6. **Novo tipo de feedback**: `sales_client_indecision`
7. **Heurística completa** de detecção

**Serviço Python - Opcional (melhorias)**:
8. **Flags específicas de indecisão** no `_generate_semantic_flags()`
9. **Melhoria na detecção de keywords condicionais**
10. **Métricas específicas de indecisão** pré-calculadas

### Dados já disponíveis (não precisam ser adicionados)

✅ `sales_category_aggregated` (categoria dominante, distribuição, estabilidade)  
✅ `sales_category_trend` (tendência, força, velocidade)  
✅ `sales_category_transition` (transições laterais)  
✅ `sales_category_ambiguity` (linguagem condicional)  
✅ `sales_category` (categorias `stalling` e `objection_soft`)  
✅ `keywords` (palavras de hesitação)

### Dados que precisam ser adicionados

❌ Histórico de textos/frases no `ParticipantState`  
❌ Função para extrair frases representativas  
❌ Função para detectar padrões semânticos  
❌ Função para calcular consistência temporal  
❌ Função para calcular confidence combinado

---

## 🎯 Fases de Implementação

> **Nota**: As Fases 1-7 são **obrigatórias** e já foram implementadas no backend.  
> As Fases 8-10 são **opcionais** e melhoram a precisão da detecção no serviço Python.

### **Fase 1: Armazenamento de Histórico de Textos**

**Objetivo**: Armazenar histórico de textos analisados para permitir extração de frases representativas.

**Arquivos a modificar**:
- `apps/backend/src/feedback/a2e2/types.ts` - Adicionar `textHistory` ao `ParticipantState`
- `apps/backend/src/feedback/feedback.aggregator.service.ts` - Atualizar `updateStateWithTextAnalysis()`

**Implementação**:

1. **Adicionar tipo `TextHistoryEntry` em `types.ts`**:
```typescript
export interface TextHistoryEntry {
  text: string;
  timestamp: number;
  sales_category?: string | null;
  sales_category_confidence?: number | null;
  sales_category_intensity?: number | null;
  sales_category_ambiguity?: number | null;
}
```

2. **Adicionar `textHistory` ao `ParticipantState.textAnalysis`**:
```typescript
textAnalysis: {
  // ... campos existentes
  textHistory?: TextHistoryEntry[];  // Últimos N textos (padrão: 20)
}
```

3. **Atualizar `updateStateWithTextAnalysis()` para manter histórico**:
```typescript
private updateStateWithTextAnalysis(
  state: ParticipantState,
  evt: TextAnalysisResult,
): void {
  // ... código existente
  
  // Manter histórico de textos (últimos 20)
  const maxHistorySize = 20;
  const historyEntry: TextHistoryEntry = {
    text: evt.text,
    timestamp: evt.timestamp,
    sales_category: evt.analysis.sales_category ?? null,
    sales_category_confidence: evt.analysis.sales_category_confidence ?? null,
    sales_category_intensity: evt.analysis.sales_category_intensity ?? null,
    sales_category_ambiguity: evt.analysis.sales_category_ambiguity ?? null,
  };
  
  state.textAnalysis.textHistory = state.textAnalysis.textHistory ?? [];
  state.textAnalysis.textHistory.push(historyEntry);
  
  // Manter apenas últimos N textos
  if (state.textAnalysis.textHistory.length > maxHistorySize) {
    state.textAnalysis.textHistory = state.textAnalysis.textHistory.slice(-maxHistorySize);
  }
}
```

**Critérios de aceitação**:
- [ ] Histórico armazena últimos 20 textos
- [ ] Cada entrada contém texto, timestamp e campos de sales_category
- [ ] Histórico é automaticamente limitado a 20 entradas
- [ ] Histórico persiste entre múltiplas análises

**Tempo estimado**: 30 minutos

---

### **Fase 2: Função para Extrair Frases Representativas**

**Objetivo**: Extrair frases do histórico que representam padrões de indecisão.

**Arquivos a criar/modificar**:
- `apps/backend/src/feedback/feedback.aggregator.service.ts` - Adicionar método `extractRepresentativePhrases()`

**Implementação**:

```typescript
/**
 * Extrai frases representativas de indecisão do histórico de textos.
 * 
 * Filtra textos que:
 * - Têm categoria de indecisão (stalling, objection_soft)
 * - Têm confiança mínima (>= 0.6)
 * - Estão dentro da janela temporal especificada
 * 
 * Retorna até maxPhrases frases, ordenadas por confiança (maior primeiro).
 */
private extractRepresentativePhrases(
  state: ParticipantState,
  now: number,
  windowMs: number = 60000, // Últimos 60 segundos
  maxPhrases: number = 5,
  minConfidence: number = 0.6
): string[] {
  const textHistory = state.textAnalysis?.textHistory ?? [];
  if (textHistory.length === 0) {
    return [];
  }
  
  const cutoffTime = now - windowMs;
  const indecisionCategories = ['stalling', 'objection_soft'];
  
  // Filtrar textos de indecisão dentro da janela temporal
  const indecisionTexts = textHistory
    .filter(entry => {
      // Verificar timestamp
      if (entry.timestamp < cutoffTime) {
        return false;
      }
      
      // Verificar categoria
      if (!entry.sales_category || !indecisionCategories.includes(entry.sales_category)) {
        return false;
      }
      
      // Verificar confiança mínima
      if ((entry.sales_category_confidence ?? 0) < minConfidence) {
        return false;
      }
      
      return true;
    })
    // Ordenar por confiança (maior primeiro)
    .sort((a, b) => (b.sales_category_confidence ?? 0) - (a.sales_category_confidence ?? 0))
    // Limitar quantidade
    .slice(0, maxPhrases)
    // Extrair apenas o texto
    .map(entry => entry.text);
  
  return indecisionTexts;
}
```

**Critérios de aceitação**:
- [ ] Retorna até 5 frases representativas
- [ ] Filtra apenas categorias de indecisão (stalling, objection_soft)
- [ ] Filtra por confiança mínima (>= 0.6)
- [ ] Filtra por janela temporal (últimos 60s)
- [ ] Ordena por confiança (maior primeiro)
- [ ] Retorna array vazio se não houver textos válidos

**Tempo estimado**: 20 minutos

---

### **Fase 3: Função para Detectar Padrões Semânticos**

**Objetivo**: Detectar três padrões específicos de indecisão.

**Arquivos a criar/modificar**:
- `apps/backend/src/feedback/feedback.aggregator.service.ts` - Adicionar método `detectIndecisionPatterns()`

**Implementação**:

```typescript
/**
 * Detecta padrões semânticos de indecisão baseado em análise contextual.
 * 
 * Padrões detectados:
 * 1. decision_postponement: Cliente consistentemente posterga decisões
 * 2. conditional_language: Cliente usa linguagem condicional/aberta
 * 3. lack_of_commitment: Cliente evita compromissos claros
 */
private detectIndecisionPatterns(
  state: ParticipantState
): {
  decision_postponement: boolean;
  conditional_language: boolean;
  lack_of_commitment: boolean;
} {
  const textAnalysis = state.textAnalysis;
  if (!textAnalysis) {
    return {
      decision_postponement: false,
      conditional_language: false,
      lack_of_commitment: false,
    };
  }
  
  const aggregated = textAnalysis.sales_category_aggregated;
  const trend = textAnalysis.sales_category_trend;
  const ambiguity = textAnalysis.sales_category_ambiguity ?? 0;
  const keywords = textAnalysis.keywords ?? [];
  
  // Padrão 1: Decision Postponement
  // Cliente consistentemente posterga decisões
  // Requisitos:
  // - Categoria dominante é stalling
  // - Tendência estável (sem progresso)
  // - Velocidade próxima de zero
  const isStallingDominant = aggregated?.dominant_category === 'stalling';
  const isStable = trend?.trend === 'stable';
  const isLowVelocity = (trend?.velocity ?? 1) < 0.1;
  const decision_postponement = isStallingDominant && isStable && isLowVelocity;
  
  // Padrão 2: Conditional Language
  // Cliente usa linguagem condicional/aberta
  // Requisitos:
  // - Alta ambiguidade semântica (> 0.7)
  // - Keywords condicionais presentes
  const conditionalKeywords = [
    'talvez', 'pensar', 'avaliar', 'depois', 'ver', 'consultar',
    'depende', 'preciso', 'vou ver', 'deixa', 'analisar'
  ];
  const hasConditionalKeywords = keywords.some(kw => 
    conditionalKeywords.some(ck => kw.toLowerCase().includes(ck))
  );
  const conditional_language = ambiguity > 0.7 && hasConditionalKeywords;
  
  // Padrão 3: Lack of Commitment
  // Cliente evita compromissos claros
  // Requisitos:
  // - Baixa estabilidade (< 0.5) = alterna entre categorias
  // - Alta proporção de categorias de indecisão (> 60%)
  const stability = aggregated?.stability ?? 0;
  const distribution = aggregated?.category_distribution ?? {};
  const indecisionRatio = (distribution.stalling ?? 0) + (distribution.objection_soft ?? 0);
  const lack_of_commitment = stability < 0.5 && indecisionRatio > 0.6;
  
  return {
    decision_postponement,
    conditional_language,
    lack_of_commitment,
  };
}
```

**Critérios de aceitação**:
- [ ] Detecta `decision_postponement` quando stalling dominante + estável
- [ ] Detecta `conditional_language` quando alta ambiguidade + keywords condicionais
- [ ] Detecta `lack_of_commitment` quando baixa estabilidade + alta proporção de indecisão
- [ ] Retorna objeto com três flags booleanas
- [ ] Retorna todos false se textAnalysis não existir

**Tempo estimado**: 30 minutos

---

### **Fase 4: Função para Calcular Consistência Temporal**

**Objetivo**: Verificar se o padrão de indecisão se mantém consistente ao longo do tempo.

**Arquivos a criar/modificar**:
- `apps/backend/src/feedback/feedback.aggregator.service.ts` - Adicionar método `calculateTemporalConsistency()`

**Implementação**:

```typescript
/**
 * Calcula consistência temporal do padrão de indecisão.
 * 
 * Verifica se o padrão se mantém consistente ao longo de uma janela temporal.
 * 
 * Requisitos para consistência:
 * - Padrão presente em pelo menos 70% dos chunks na janela
 * - Estabilidade da categoria dominante > 0.5
 * - Tendência permanece estável ao longo do tempo
 */
private calculateTemporalConsistency(
  state: ParticipantState,
  now: number,
  windowMs: number = 60000 // Últimos 60 segundos
): boolean {
  const textAnalysis = state.textAnalysis;
  if (!textAnalysis) {
    return false;
  }
  
  const textHistory = textAnalysis.textHistory ?? [];
  if (textHistory.length === 0) {
    return false;
  }
  
  const cutoffTime = now - windowMs;
  const indecisionCategories = ['stalling', 'objection_soft'];
  
  // Filtrar textos dentro da janela temporal
  const windowTexts = textHistory.filter(entry => entry.timestamp >= cutoffTime);
  if (windowTexts.length === 0) {
    return false;
  }
  
  // Contar textos com categoria de indecisão
  const indecisionTexts = windowTexts.filter(entry => 
    entry.sales_category && 
    indecisionCategories.includes(entry.sales_category) &&
    (entry.sales_category_confidence ?? 0) >= 0.6
  );
  
  // Verificar proporção mínima (70%)
  const indecisionRatio = indecisionTexts.length / windowTexts.length;
  if (indecisionRatio < 0.7) {
    return false;
  }
  
  // Verificar estabilidade da categoria dominante
  const aggregated = textAnalysis.sales_category_aggregated;
  const stability = aggregated?.stability ?? 0;
  if (stability < 0.5) {
    return false;
  }
  
  // Verificar tendência estável
  const trend = textAnalysis.sales_category_trend;
  const isStable = trend?.trend === 'stable';
  
  return isStable;
}
```

**Critérios de aceitação**:
- [ ] Retorna true se padrão presente em >= 70% dos chunks na janela
- [ ] Verifica estabilidade da categoria dominante (>= 0.5)
- [ ] Verifica tendência estável
- [ ] Retorna false se não houver dados suficientes
- [ ] Considera apenas textos com confiança >= 0.6

**Tempo estimado**: 25 minutos

---

### **Fase 5: Função para Calcular Confidence Combinado**

**Objetivo**: Calcular confidence combinando múltiplos sinais de indecisão.

**Arquivos a criar/modificar**:
- `apps/backend/src/feedback/feedback.aggregator.service.ts` - Adicionar método `calculateIndecisionConfidence()`

**Implementação**:

```typescript
/**
 * Calcula confidence combinado para detecção de indecisão.
 * 
 * Combina múltiplos sinais:
 * - Estabilidade da categoria dominante
 * - Força da tendência
 * - Volume de dados (total_chunks)
 * - Proporção de categorias de indecisão
 * - Consistência temporal
 * 
 * Retorna valor de 0.0 a 1.0.
 */
private calculateIndecisionConfidence(
  state: ParticipantState,
  patterns: {
    decision_postponement: boolean;
    conditional_language: boolean;
    lack_of_commitment: boolean;
  },
  temporalConsistency: boolean
): number {
  const textAnalysis = state.textAnalysis;
  if (!textAnalysis) {
    return 0.0;
  }
  
  const aggregated = textAnalysis.sales_category_aggregated;
  const trend = textAnalysis.sales_category_trend;
  
  // Base: número de padrões detectados (0 a 3)
  const patternsCount = Object.values(patterns).filter(Boolean).length;
  const patternsScore = patternsCount / 3.0; // 0.0 a 1.0
  
  // Estabilidade da categoria dominante (0.0 a 1.0)
  const stability = aggregated?.stability ?? 0;
  
  // Força da tendência (0.0 a 1.0)
  const trendStrength = trend?.trend_strength ?? 0;
  
  // Volume de dados (normalizado, 0.0 a 1.0)
  // Mínimo 5 chunks, ideal 10+ chunks
  const totalChunks = aggregated?.chunks_with_category ?? 0;
  const volumeScore = Math.min(1.0, totalChunks / 10.0);
  
  // Proporção de categorias de indecisão (0.0 a 1.0)
  const distribution = aggregated?.category_distribution ?? {};
  const indecisionRatio = (distribution.stalling ?? 0) + (distribution.objection_soft ?? 0);
  
  // Consistência temporal (0.0 ou 1.0)
  const consistencyScore = temporalConsistency ? 1.0 : 0.0;
  
  // Calcular confidence combinado (média ponderada)
  // Pesos:
  // - Padrões detectados: 30%
  // - Estabilidade: 20%
  // - Força da tendência: 15%
  // - Volume de dados: 15%
  // - Proporção de indecisão: 10%
  // - Consistência temporal: 10%
  const confidence = (
    patternsScore * 0.30 +
    stability * 0.20 +
    trendStrength * 0.15 +
    volumeScore * 0.15 +
    indecisionRatio * 0.10 +
    consistencyScore * 0.10
  );
  
  // Garantir range [0, 1]
  return Math.max(0.0, Math.min(1.0, confidence));
}
```

**Critérios de aceitação**:
- [ ] Combina múltiplos sinais com pesos apropriados
- [ ] Retorna valor entre 0.0 e 1.0
- [ ] Considera padrões detectados, estabilidade, tendência, volume, proporção e consistência
- [ ] Retorna 0.0 se textAnalysis não existir

**Tempo estimado**: 25 minutos

---

### **Fase 6: Adicionar Novo Tipo de Feedback**

**Objetivo**: Adicionar `sales_client_indecision` ao enum de tipos de feedback.

**Arquivos a modificar**:
- `apps/backend/src/feedback/feedback.types.ts` - Adicionar ao enum `type`

**Implementação**:

```typescript
export interface FeedbackEventPayload {
  id: string;
  type:
    | 'volume_baixo'
    | 'volume_alto'
    | // ... tipos existentes ...
    | 'sales_price_window_open'
    | 'sales_decision_signal'
    | 'sales_ready_to_close'
    | 'sales_objection_escalating'
    | 'sales_conversation_stalling'
    | 'sales_category_transition'
    | 'sales_client_indecision';  // ← NOVO
  // ... resto da interface
}
```

**Critérios de aceitação**:
- [ ] `sales_client_indecision` adicionado ao enum
- [ ] TypeScript compila sem erros
- [ ] Tipo é reconhecido em todos os lugares que usam o enum

**Tempo estimado**: 5 minutos

---

### **Fase 7: Implementar Heurística Completa de Detecção**

**Objetivo**: Implementar função completa que detecta indecisão e gera feedback.

**Arquivos a criar/modificar**:
- `apps/backend/src/feedback/feedback.aggregator.service.ts` - Adicionar método `detectClientIndecision()`

**Implementação**:

```typescript
/**
 * Detecta padrão consistente de indecisão do cliente.
 * 
 * Características detectadas:
 * - Postergar decisões
 * - Solicitar mais tempo ou validações
 * - Repetir dúvidas semelhantes
 * - Evitar compromissos claros
 * - Usar linguagem condicional ou aberta
 */
private detectClientIndecision(
  state: ParticipantState,
  evt: TextAnalysisResult,
  now: number,
): FeedbackEventPayload | null {
  const textAnalysis = state.textAnalysis;
  if (!textAnalysis) {
    return null;
  }
  
  // Verificar cooldown (2 minutos)
  if (this.inCooldown(state, 'sales_client_indecision', now)) {
    return null;
  }
  
  // Verificar volume mínimo de dados
  const aggregated = textAnalysis.sales_category_aggregated;
  const hasEnoughData = (aggregated?.chunks_with_category ?? 0) >= 5;
  if (!hasEnoughData) {
    return null;
  }
  
  // Detectar padrões semânticos
  const patterns = this.detectIndecisionPatterns(state);
  
  // Verificar se pelo menos um padrão foi detectado
  const hasPattern = Object.values(patterns).some(Boolean);
  if (!hasPattern) {
    return null;
  }
  
  // Calcular consistência temporal
  const temporalConsistency = this.calculateTemporalConsistency(state, now, 60000);
  
  // Calcular confidence combinado
  const confidence = this.calculateIndecisionConfidence(state, patterns, temporalConsistency);
  
  // Threshold mínimo de confidence (0.7)
  if (confidence < 0.7) {
    return null;
  }
  
  // Extrair frases representativas
  const representativePhrases = this.extractRepresentativePhrases(
    state,
    now,
    60000, // Últimos 60s
    5,     // Máximo 5 frases
    0.6    // Confiança mínima
  );
  
  // Se não houver frases representativas, não gerar feedback
  if (representativePhrases.length === 0) {
    return null;
  }
  
  // Construir lista de padrões detectados
  const patternsDetected = Object.entries(patterns)
    .filter(([, detected]) => detected)
    .map(([pattern]) => pattern);
  
  // Construir mensagem
  const message = temporalConsistency
    ? 'O cliente repete padrões de adiamento e evita compromissos claros ao longo da conversa.'
    : 'Padrões de indecisão detectados na conversa recente.';
  
  // Construir tips
  const tips = [
    `Padrões detectados: ${patternsDetected.join(', ')}`,
    `Frases representativas: ${representativePhrases.slice(0, 3).map(p => `"${p}"`).join(', ')}`,
    temporalConsistency
      ? 'Consistência temporal: padrão mantido ao longo da conversa'
      : 'Consistência temporal: padrão detectado recentemente',
  ];
  
  // Gerar feedback
  const window = this.window(state, now, 60000); // Últimos 60s
  this.setCooldown(state, 'sales_client_indecision', now, 120000); // Cooldown de 2min
  
  return {
    id: this.makeId(),
    type: 'sales_client_indecision',
    severity: 'warning',
    ts: now,
    meetingId: evt.meetingId,
    participantId: evt.participantId,
    participantName: this.index.getParticipantName(evt.meetingId, evt.participantId) ?? undefined,
    window: { start: window.start, end: window.end },
    message,
    tips,
    metadata: {
      confidence: Math.round(confidence * 100) / 100, // Arredondar para 2 casas
      semantic_patterns_detected: patternsDetected,
      representative_phrases: representativePhrases,
      temporal_consistency: temporalConsistency,
      sales_category: textAnalysis.sales_category ?? undefined,
      sales_category_confidence: textAnalysis.sales_category_confidence ?? undefined,
      sales_category_aggregated: aggregated ?? undefined,
    },
  };
}
```

**Integração no fluxo existente**:

Adicionar chamada em `handleTextAnalysis()`:

```typescript
@OnEvent('text_analysis_result', { async: true })
handleTextAnalysis(evt: TextAnalysisResult): void {
  // ... código existente ...
  
  // Detecção de indecisão do cliente
  const indecisionFeedback = this.detectClientIndecision(state, evt, now);
  if (indecisionFeedback) {
    this.delivery.publishToHosts(evt.meetingId, indecisionFeedback);
  }
}
```

**Critérios de aceitação**:
- [ ] Detecta indecisão quando padrões são consistentes
- [ ] Gera feedback apenas com confidence >= 0.7
- [ ] Inclui frases representativas no metadata
- [ ] Inclui padrões detectados no metadata
- [ ] Respeita cooldown de 2 minutos
- [ ] Requer mínimo de 5 chunks com categoria
- [ ] Integrado no fluxo de `handleTextAnalysis()`

**Tempo estimado**: 45 minutos

---

## 📋 Checklist de Implementação

### Fase 1: Armazenamento de Histórico
- [ ] Adicionar `TextHistoryEntry` interface
- [ ] Adicionar `textHistory` ao `ParticipantState`
- [ ] Atualizar `updateStateWithTextAnalysis()`
- [ ] Testar armazenamento de histórico

### Fase 2: Extração de Frases
- [ ] Implementar `extractRepresentativePhrases()`
- [ ] Testar filtragem por categoria
- [ ] Testar filtragem por confiança
- [ ] Testar ordenação por confiança

### Fase 3: Detecção de Padrões
- [ ] Implementar `detectIndecisionPatterns()`
- [ ] Testar detecção de `decision_postponement`
- [ ] Testar detecção de `conditional_language`
- [ ] Testar detecção de `lack_of_commitment`

### Fase 4: Consistência Temporal
- [ ] Implementar `calculateTemporalConsistency()`
- [ ] Testar verificação de proporção (70%)
- [ ] Testar verificação de estabilidade
- [ ] Testar verificação de tendência

### Fase 5: Confidence Combinado
- [ ] Implementar `calculateIndecisionConfidence()`
- [ ] Testar combinação de sinais
- [ ] Testar pesos apropriados
- [ ] Testar range [0, 1]

### Fase 6: Novo Tipo de Feedback
- [ ] Adicionar `sales_client_indecision` ao enum
- [ ] Verificar compilação TypeScript
- [ ] Verificar uso em outros lugares

### Fase 7: Heurística Completa
- [ ] Implementar `detectClientIndecision()`
- [ ] Integrar no fluxo `handleTextAnalysis()`
- [ ] Testar geração de feedback
- [ ] Testar cooldown
- [ ] Testar threshold de confidence

---

## ⏱️ Estimativa Total

| Fase | Tempo Estimado |
|------|----------------|
| Fase 1: Armazenamento de Histórico | 30 min |
| Fase 2: Extração de Frases | 20 min |
| Fase 3: Detecção de Padrões | 30 min |
| Fase 4: Consistência Temporal | 25 min |
| Fase 5: Confidence Combinado | 25 min |
| Fase 6: Novo Tipo de Feedback | 5 min |
| Fase 7: Heurística Completa | 45 min |
| **TOTAL** | **~3 horas** |

---

## 🧪 Testes Sugeridos

### Testes Unitários

1. **Teste de armazenamento de histórico**:
   - Verificar que histórico mantém últimos 20 textos
   - Verificar que histórico é limitado corretamente

2. **Teste de extração de frases**:
   - Verificar filtragem por categoria
   - Verificar filtragem por confiança
   - Verificar ordenação por confiança

3. **Teste de detecção de padrões**:
   - Verificar cada padrão individualmente
   - Verificar combinação de padrões

4. **Teste de consistência temporal**:
   - Verificar com dados suficientes
   - Verificar com dados insuficientes
   - Verificar com padrão inconsistente

5. **Teste de confidence**:
   - Verificar cálculo com diferentes combinações
   - Verificar range [0, 1]

### Testes de Integração

1. **Teste de geração de feedback**:
   - Verificar que feedback é gerado quando condições são atendidas
   - Verificar que feedback não é gerado quando condições não são atendidas
   - Verificar estrutura do feedback gerado

2. **Teste de cooldown**:
   - Verificar que cooldown é respeitado
   - Verificar que feedback pode ser gerado após cooldown

---

## 📝 Notas de Implementação

### Considerações de Performance

- Histórico limitado a 20 textos para evitar crescimento excessivo de memória
- Cálculos são O(n) onde n é o tamanho do histórico
- Cooldown de 2 minutos evita spam de feedbacks

### Considerações de Precisão

- Threshold de confidence (0.7) pode ser ajustado baseado em dados reais
- Proporção mínima de 70% para consistência temporal pode ser ajustada
- Pesos do cálculo de confidence podem ser refinados

### Melhorias Futuras

- Adicionar métricas de qualidade da detecção
- Permitir ajuste de thresholds via configuração
- Adicionar mais padrões semânticos se necessário
- Melhorar extração de frases representativas (ex: usar embeddings)

---

## ✅ Critérios de Sucesso

A implementação será considerada bem-sucedida quando:

1. ✅ Histórico de textos é mantido corretamente
2. ✅ Frases representativas são extraídas corretamente
3. ✅ Padrões semânticos são detectados corretamente
4. ✅ Consistência temporal é calculada corretamente
5. ✅ Confidence é calculado corretamente
6. ✅ Feedback é gerado quando condições são atendidas
7. ✅ Feedback não é gerado quando condições não são atendidas
8. ✅ Cooldown é respeitado
9. ✅ Estrutura do feedback está correta
10. ✅ Testes passam

---

## 🐍 Melhorias no Serviço Python (Opcional)

Embora o backend já tenha todos os dados necessários para detectar indecisão, podemos melhorar o serviço Python para facilitar e tornar a detecção mais precisa. Estas melhorias são **opcionais** mas recomendadas.

### **Fase 8 (Opcional): Adicionar Flags de Indecisão no Python**

**Objetivo**: Adicionar flags específicas de indecisão no método `_generate_semantic_flags()` para facilitar detecção no backend.

**Arquivos a modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py` - Adicionar flags de indecisão
- `apps/text-analysis/src/services/analysis_service.py` - Garantir que flags são retornadas
- `apps/backend/src/pipeline/text-analysis.service.ts` - Adicionar campos de flags de indecisão

**Implementação**:

1. **Adicionar flags de indecisão em `_generate_semantic_flags()`**:

```python
def _generate_semantic_flags(
    self,
    category: Optional[str],
    confidence: float,
    intensity: float,
    ambiguity: float
) -> Dict[str, bool]:
    # ... flags existentes ...
    
    # Flag: Indecisão detectada
    # Indica que há sinais de indecisão no texto atual
    # Requisitos: categoria stalling ou objection_soft + alta ambiguidade ou baixa confiança
    flags['indecision_detected'] = (
        category in ['stalling', 'objection_soft'] and
        (ambiguity > 0.6 or confidence < 0.7)
    )
    
    # Flag: Postergação de decisão
    # Indica que cliente está postergando decisão
    # Requisitos: categoria stalling + alta confiança + baixa intensidade
    flags['decision_postponement_signal'] = (
        category == 'stalling' and
        confidence > 0.7 and
        intensity < 0.7  # Intensidade baixa = hesitação
    )
    
    # Flag: Linguagem condicional
    # Indica uso de linguagem condicional/aberta
    # Requisitos: alta ambiguidade + categoria de indecisão
    flags['conditional_language_signal'] = (
        category in ['stalling', 'objection_soft'] and
        ambiguity > 0.7
    )
    
    return flags
```

2. **Atualizar interface TypeScript para incluir novas flags**:

```typescript
sales_category_flags?: {
  price_window_open?: boolean;
  decision_signal_strong?: boolean;
  ready_to_close?: boolean;
  // Novas flags de indecisão
  indecision_detected?: boolean;
  decision_postponement_signal?: boolean;
  conditional_language_signal?: boolean;
} | null;
```

**Benefícios**:
- Backend pode usar flags diretamente sem recalcular
- Detecção mais rápida e eficiente
- Flags podem ser usadas em outras heurísticas

**Tempo estimado**: 30 minutos

---

### **Fase 9 (Opcional): Melhorar Detecção de Keywords Condicionais**

**Objetivo**: Expandir lista de keywords condicionais e melhorar detecção no Python.

**Arquivos a modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py` - Adicionar lista expandida de keywords
- `apps/text-analysis/src/services/analysis_service.py` - Adicionar campo `conditional_keywords_detected`

**Implementação**:

1. **Adicionar constante de keywords condicionais**:

```python
# Em bert_analyzer.py
CONDITIONAL_KEYWORDS = [
    'talvez', 'pensar', 'avaliar', 'depois', 'ver', 'consultar',
    'depende', 'preciso', 'vou ver', 'deixa', 'analisar',
    'considerar', 'refletir', 'avaliar melhor', 'pensar melhor',
    'preciso pensar', 'vou considerar', 'deixa eu ver',
    'não tenho certeza', 'não sei', 'talvez depois',
    'preciso avaliar', 'vou analisar', 'deixa eu pensar',
    'não tenho pressa', 'sem pressa', 'depois eu vejo'
]

def detect_conditional_keywords(self, text: str, keywords: List[str]) -> List[str]:
    """
    Detecta keywords condicionais no texto.
    
    Retorna lista de keywords condicionais encontradas.
    """
    text_lower = text.lower()
    detected = []
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            detected.append(keyword)
    
    # Verificar também na lista de keywords extraídas
    for kw in keywords:
        for conditional in CONDITIONAL_KEYWORDS:
            if conditional in kw.lower():
                if conditional not in detected:
                    detected.append(conditional)
    
    return detected
```

2. **Adicionar campo no resultado**:

```python
# Em analysis_service.py
conditional_keywords = analyzer.detect_conditional_keywords(chunk.text, keywords)

result = {
    # ... campos existentes ...
    'conditional_keywords_detected': conditional_keywords,
}
```

**Benefícios**:
- Detecção mais precisa de linguagem condicional
- Lista centralizada e reutilizável
- Pode ser expandida facilmente

**Tempo estimado**: 20 minutos

---

### **Fase 10 (Opcional): Adicionar Métricas de Indecisão**

**Objetivo**: Calcular métricas específicas de indecisão no Python para facilitar análise no backend.

**Arquivos a modificar**:
- `apps/text-analysis/src/models/bert_analyzer.py` - Adicionar método `calculate_indecision_metrics()`
- `apps/text-analysis/src/services/analysis_service.py` - Incluir métricas no resultado

**Implementação**:

```python
def calculate_indecision_metrics(
    self,
    category: Optional[str],
    confidence: float,
    intensity: float,
    ambiguity: float,
    conditional_keywords: List[str]
) -> Dict[str, Any]:
    """
    Calcula métricas específicas de indecisão.
    
    Returns:
        Dict com métricas:
        - indecision_score: float (0.0 a 1.0)
        - postponement_likelihood: float (0.0 a 1.0)
        - conditional_language_score: float (0.0 a 1.0)
    """
    metrics = {
        'indecision_score': 0.0,
        'postponement_likelihood': 0.0,
        'conditional_language_score': 0.0,
    }
    
    # Score geral de indecisão
    if category in ['stalling', 'objection_soft']:
        # Baseado em categoria, ambiguidade e confiança
        base_score = 0.5 if category == 'stalling' else 0.3
        ambiguity_boost = ambiguity * 0.3
        confidence_penalty = (1.0 - confidence) * 0.2
        metrics['indecision_score'] = min(1.0, base_score + ambiguity_boost + confidence_penalty)
    
    # Probabilidade de postergação
    if category == 'stalling':
        metrics['postponement_likelihood'] = min(1.0, confidence * intensity)
    
    # Score de linguagem condicional
    if conditional_keywords:
        metrics['conditional_language_score'] = min(1.0, len(conditional_keywords) / 5.0)
    metrics['conditional_language_score'] = max(
        metrics['conditional_language_score'],
        ambiguity * 0.5
    )
    
    return metrics
```

**Benefícios**:
- Métricas pré-calculadas facilitam heurísticas no backend
- Reduz processamento no backend
- Métricas podem ser usadas para outros propósitos

**Tempo estimado**: 30 minutos

---

### **Resumo das Melhorias Opcionais no Python**

| Fase | Descrição | Tempo | Prioridade |
|------|-----------|-------|------------|
| Fase 8 | Flags de indecisão | 30 min | Média |
| Fase 9 | Keywords condicionais | 20 min | Baixa |
| Fase 10 | Métricas de indecisão | 30 min | Média |
| **TOTAL** | | **~1.5 horas** | |

**Nota**: Estas melhorias são opcionais porque o backend já tem todos os dados necessários. Elas facilitam a detecção mas não são obrigatórias para o funcionamento básico.

---

**Próximos Passos**: 
- **Backend**: Começar pela Fase 1 e seguir sequencialmente até Fase 7 (✅ COMPLETO)
- **Python (Opcional)**: Implementar Fases 8-10 se desejar melhorar precisão e facilitar detecção

