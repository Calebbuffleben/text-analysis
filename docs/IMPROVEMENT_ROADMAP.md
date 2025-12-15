# Roadmap de Melhorias - Sistema de Análise Semântica para Vendas

**Versão**: 1.0  
**Data**: 2025-01-XX  
**Autor**: Arquitetura de Software / ML Engineering

---

## 1️⃣ Visão Geral do Planejamento

### Objetivo do Plano

Evoluir o sistema de análise semântica para reuniões de vendas, aumentando robustez, precisão e qualidade dos feedbacks gerados, mantendo simplicidade arquitetural e performance adequada.

**Meta Principal**: Gerar feedbacks mais confiáveis e acionáveis para vendedores, como:
- "Agora é o momento de falar sobre preço"
- "Cliente demonstrando objeção - requer abordagem diferente"
- "Cliente pronto para avançar - acelerar fechamento"

### Problemas Atuais que Motivam as Melhorias

**Limitações Observadas**:

1. **Contexto Curto**: Classificação baseada apenas no chunk atual, sem histórico
   - Não detecta transições de estágio (ex: value_exploration → price_interest)
   - Não identifica padrões ao longo da conversa
   - Pode gerar feedbacks contraditórios em sequência

2. **Decisões Pontuais**: Cada chunk é analisado isoladamente
   - Ruído de frases isoladas pode gerar falsos positivos
   - Falta agregação temporal para reduzir instabilidade
   - Não considera tendência semântica

3. **Confiança Limitada**: Threshold fixo (0.3) pode ser muito permissivo
   - Textos ambíguos podem ser classificados incorretamente
   - Falta métrica de ambiguidade semântica
   - Não diferencia entre alta confiança e baixa confiança

4. **Sinais Semânticos Limitados**: Apenas categoria + confiança
   - Não indica intensidade do sinal
   - Não indica direção da conversa (avançando/estagnada/regredindo)
   - Não fornece flags semânticas específicas para heurísticas

5. **Falta de Observabilidade**: Dificuldade em entender por que um feedback foi gerado
   - Logs não explicam decisões semânticas
   - Métricas de qualidade não são coletadas
   - Validação manual é trabalhosa

### Princípios de Design

**Simplicidade**: Manter arquitetura simples e explicável
- Evitar over-engineering
- Preferir soluções incrementais
- Manter separação clara de responsabilidades

**Performance**: Latência aceitável para tempo quase real
- Primeira análise: < 1s (aceitável)
- Análises subsequentes: < 100ms (ideal)
- Cache agressivo quando possível

**Modularidade**: Componentes desacoplados e testáveis
- Python retorna sinais semânticos estruturados
- Backend decide feedbacks baseado em heurísticas
- Fácil adicionar novos sinais sem quebrar existentes

**Baixo Custo**: Operar eficientemente sem GPU dedicada
- CPU-first com possibilidade futura de GPU
- Modelos leves quando possível
- Cache inteligente para reduzir recálculos

---

## 2️⃣ Diagnóstico da Arquitetura Atual

### Pontos Fortes da Implementação Atual

✅ **Base Sólida**:
- SBERT multilíngue bem escolhido (paraphrase-multilingual-MiniLM-L12-v2)
- Cache de embeddings dos exemplos funciona bem
- Lazy loading implementado corretamente
- Tratamento de erros gracioso (não bloqueia outras análises)

✅ **Separação de Responsabilidades**:
- Python foca em análise semântica
- Backend foca em heurísticas de negócio
- Interfaces bem definidas

✅ **Performance Adequada**:
- Primeira chamada: ~400-500ms (aceitável)
- Chamadas subsequentes: ~5ms (excelente)
- Memória: ~30KB adicional (negligível)

✅ **Cobertura de Categorias**:
- 8 categorias bem definidas
- 80 exemplos de referência (10 por categoria)
- Cobertura adequada de variações linguísticas

### Limitações Observadas

**1. Análise Semântica Isolada**

**Problema**: Cada chunk é analisado independentemente, sem contexto histórico.

**Impacto**:
- Não detecta progressão: `value_exploration` → `price_interest` → `decision_signal`
- Não identifica regressão: `decision_signal` → `objection_soft` → `objection_hard`
- Pode gerar feedbacks contraditórios em sequência

**Exemplo**:
```
Chunk 1: "Como isso funciona?" → value_exploration
Chunk 2: "Quanto custa?" → price_interest
Chunk 3: "Preciso pensar" → stalling
```
Sistema atual: Três classificações isoladas  
Sistema ideal: Detecta progressão → regressão

**2. Falta de Agregação Temporal**

**Problema**: Ruído de frases isoladas pode gerar falsos positivos.

**Impacto**:
- Frase ambígua pode ser classificada incorretamente
- Feedback prematuro pode ser gerado
- Instabilidade em classificações consecutivas

**Exemplo**:
```
Chunk 1: "Não sei" → objection_soft (falso positivo)
Chunk 2: "Mas me interessa" → value_exploration
Chunk 3: "Quanto custa?" → price_interest
```
Sistema atual: Gera feedback de objeção no chunk 1  
Sistema ideal: Agrega contexto e ignora ruído

**3. Sinais Semânticos Limitados**

**Problema**: Apenas categoria + confiança não fornece informação suficiente.

**Impacto**:
- Backend não sabe intensidade do sinal
- Não sabe direção da conversa
- Não tem flags específicas para heurísticas

**Exemplo Atual**:
```json
{
  "sales_category": "price_interest",
  "sales_category_confidence": 0.85
}
```

**Exemplo Ideal**:
```json
{
  "sales_category": "price_interest",
  "sales_category_confidence": 0.85,
  "intensity": 0.92,
  "ambiguity": 0.15,
  "trend": "advancing",
  "flags": {
    "price_window_open": true,
    "strong_signal": true
  }
}
```

**4. Falta de Métricas de Qualidade**

**Problema**: Não há visibilidade sobre qualidade das classificações.

**Impacto**:
- Dificuldade em ajustar thresholds
- Não há feedback loop para melhorar exemplos
- Validação manual trabalhosa

### Gargalos Técnicos Identificados

**CPU**:
- Modelo SBERT roda em CPU (aceitável, mas GPU seria melhor)
- Primeira classificação é custosa (~400ms)
- Batch processing não implementado

**Latência**:
- Primeira análise: ~400-500ms (aceitável)
- Análises subsequentes: ~5ms (excelente)
- **Gargalo**: Se precisar comparar com histórico, latência aumenta

**Confiança**:
- Threshold fixo (0.3) pode ser muito permissivo
- Não diferencia entre alta e baixa confiança
- Falta métrica de ambiguidade

**Escalabilidade**:
- Cache funciona bem para exemplos
- Mas não há cache de comparações históricas
- Se implementar contexto, precisa otimizar

---

## 3️⃣ Melhorias Semânticas (SBERT & NLP)

### 3.1 Evolução do Modelo SBERT

**Situação Atual**: `paraphrase-multilingual-MiniLM-L12-v2`
- Dimensão: 384
- Multilíngue: ✅
- Leve: ✅
- Performance: ✅

**Opções de Evolução**:

**Opção A: Manter Modelo Atual (Recomendado para Curto Prazo)**
- ✅ Já funciona bem
- ✅ Leve e rápido
- ✅ Multilíngue
- ✅ Não requer mudanças

**Opção B: Modelo Maior (Médio Prazo)**
- `paraphrase-multilingual-mpnet-base-v2` (768 dims)
- Maior precisão, mas mais lento
- Avaliar trade-off precisão vs latência

**Opção C: Fine-tuning (Longo Prazo)**
- Treinar em dados reais de reuniões de vendas
- Melhor precisão para domínio específico
- Requer dataset anotado

**Decisão Técnica**: Manter modelo atual por enquanto. Avaliar fine-tuning após coletar dados reais.

### 3.2 Expansão e Curadoria dos Exemplos

**Situação Atual**: 10 exemplos por categoria (80 total)

**Melhorias Propostas**:

**Curto Prazo**:
- Expandir para 15-20 exemplos por categoria
- Adicionar variações regionais (Brasil vs Portugal)
- Incluir gírias e expressões informais

**Médio Prazo**:
- Curadoria baseada em dados reais
- Remover exemplos que geram falsos positivos
- Adicionar exemplos de casos difíceis

**Estrutura Proposta**:
```python
SALES_CATEGORY_EXAMPLES = {
    'price_interest': {
        'core': [...],  # Exemplos principais (10)
        'variations': [...],  # Variações linguísticas (5)
        'edge_cases': [...]  # Casos difíceis (5)
    },
    ...
}
```

**Métricas de Qualidade**:
- Taxa de acerto por exemplo
- Exemplos que geram mais falsos positivos
- Cobertura de variações linguísticas

### 3.3 Classificação Multi-Label

**Problema Atual**: Apenas uma categoria por texto

**Proposta**: Permitir múltiplas categorias quando apropriado

**Exemplo**:
```
Texto: "Quanto custa e como funciona?"
Categorias: ['price_interest', 'information_gathering']
Scores: {'price_interest': 0.75, 'information_gathering': 0.68}
```

**Implementação**:
```python
def classify_sales_category_multi(
    self,
    text: str,
    min_confidence: float = 0.3,
    max_categories: int = 2
) -> Tuple[List[Tuple[str, float]], float, Dict[str, float]]:
    """
    Retorna múltiplas categorias quando scores são próximos.
    
    Returns:
        Lista de (categoria, score) ordenada por score
        Confiança geral
        Scores de todas as categorias
    """
```

**Critério para Multi-Label**:
- Se segunda melhor categoria tem score > 0.7 × melhor score
- E ambas acima de min_confidence
- Retornar ambas como relevantes

**Uso no Backend**:
```typescript
if (sales_categories.length > 1) {
  // Cliente está em múltiplos estágios simultaneamente
  // Ex: price_interest + information_gathering
}
```

### 3.4 Score de Ambiguidade Semântica

**Problema**: Textos ambíguos podem ser classificados incorretamente

**Solução**: Calcular métrica de ambiguidade

**Algoritmo**:
```python
def calculate_ambiguity(self, scores: Dict[str, float]) -> float:
    """
    Calcula ambiguidade baseada na distribuição dos scores.
    
    Alta ambiguidade: scores muito próximos entre categorias
    Baixa ambiguidade: uma categoria claramente dominante
    
    Returns:
        float: 0.0 (claro) a 1.0 (muito ambíguo)
    """
    if not scores:
        return 1.0
    
    sorted_scores = sorted(scores.values(), reverse=True)
    
    if len(sorted_scores) < 2:
        return 0.0
    
    # Entropia normalizada dos scores
    # Alta entropia = alta ambiguidade
    import numpy as np
    scores_array = np.array(sorted_scores)
    scores_normalized = scores_array / scores_array.sum()
    entropy = -np.sum(scores_normalized * np.log(scores_normalized + 1e-10))
    max_entropy = np.log(len(scores))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0
```

**Uso**:
- Se ambiguidade > 0.7: Não gerar feedback (muito incerto)
- Se ambiguidade < 0.3: Alta confiança, pode gerar feedback
- Logar ambiguidade para análise

### 3.5 Detecção de Transição de Estágio

**Problema**: Não detecta mudanças de categoria ao longo do tempo

**Solução**: Comparar categoria atual com histórico

**Implementação**:
```python
def detect_category_transition(
    self,
    current_category: str,
    current_score: float,
    history: List[Tuple[str, float, int]]  # (categoria, score, timestamp)
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
```

**Transições Importantes**:
- `value_exploration` → `price_interest`: Cliente progredindo
- `price_interest` → `decision_signal`: Pronto para fechar
- `decision_signal` → `objection_soft`: Regressão preocupante
- `objection_soft` → `objection_hard`: Piorando

**Uso no Backend**:
```typescript
if (transition?.transition_type === 'advancing' && 
    transition.to_category === 'price_interest') {
  // Gerar feedback: "Cliente progrediu para interesse em preço"
}
```

---

## 4️⃣ Análise de Contexto Conversacional

### 4.1 Janelas de Contexto

**Problema**: Análise isolada de chunks não captura contexto

**Solução**: Manter histórico e analisar em janelas

**Implementação no Python**:

```python
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
            'timestamp': chunk['timestamp'],
            'embedding': chunk.get('embedding')
        })
        # Manter apenas janela relevante
        self._prune_history()
    
    def get_window(self, now: int) -> List[Dict[str, Any]]:
        """Retorna chunks na janela temporal"""
        cutoff = now - self.window_duration_ms
        return [
            chunk for chunk in self.history
            if chunk['timestamp'] >= cutoff
        ][-self.window_size:]
```

**Uso no TextAnalysisService**:
```python
# Manter contexto por participante/reunião
self.conversation_contexts: Dict[str, ConversationContext] = {}

def analyze(self, chunk: TranscriptionChunk):
    # ... análise atual ...
    
    # Adicionar ao contexto
    key = f"{chunk.meetingId}:{chunk.participantId}"
    if key not in self.conversation_contexts:
        self.conversation_contexts[key] = ConversationContext()
    
    context = self.conversation_contexts[key]
    context.add_chunk({
        'text': chunk.text,
        'sales_category': sales_category,
        'sales_category_confidence': sales_category_confidence,
        'timestamp': chunk.timestamp,
        'embedding': embedding
    })
    
    # Análise com contexto
    window = context.get_window(chunk.timestamp)
    # ... usar window para análise contextual ...
```

### 4.2 Agregação Temporal de Categorias

**Problema**: Categorias isoladas podem ser ruidosas

**Solução**: Agregar categorias em janela temporal

**Algoritmo**:
```python
def aggregate_categories_temporal(
    self,
    window: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Agrega categorias em janela temporal.
    
    Returns:
        {
            'dominant_category': str,  # Categoria mais frequente
            'category_distribution': Dict[str, float],  # Distribuição
            'stability': float,  # 0.0 (instável) a 1.0 (estável)
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
    
    # Calcular distribuição
    total = sum(category_counts.values())
    distribution = {
        cat: count / total
        for cat, count in category_counts.items()
    }
    
    # Categoria dominante
    dominant = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
    
    # Estabilidade (quanto mais concentrada, mais estável)
    if distribution:
        max_prob = max(distribution.values())
        stability = max_prob  # Simplificado
    else:
        stability = 0.0
    
    return {
        'dominant_category': dominant,
        'category_distribution': distribution,
        'stability': stability
    }
```

**Uso**: Backend usa categoria agregada ao invés de categoria pontual

### 4.3 Tendência Semântica ao Longo da Conversa

**Problema**: Não identifica se conversa está progredindo ou regredindo

**Solução**: Calcular tendência baseada em sequência de categorias

**Mapeamento de Progressão**:
```python
CATEGORY_PROGRESSION = {
    'information_gathering': 1,
    'value_exploration': 2,
    'price_interest': 3,
    'decision_signal': 4,
    'closing_readiness': 5,
    'stalling': 0,  # Neutro
    'objection_soft': -1,
    'objection_hard': -2
}

def calculate_semantic_trend(
    self,
    window: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calcula tendência semântica da conversa.
    
    Returns:
        {
            'trend': 'advancing' | 'stable' | 'regressing',
            'trend_strength': float,  # 0.0 a 1.0
            'current_stage': int,  # Posição na progressão
            'velocity': float  # Mudança por minuto
        }
    """
    if len(window) < 2:
        return {'trend': 'stable', 'trend_strength': 0.0}
    
    # Mapear categorias para números
    progression_values = [
        CATEGORY_PROGRESSION.get(chunk.get('sales_category'), 0)
        for chunk in window
        if chunk.get('sales_category')
    ]
    
    if len(progression_values) < 2:
        return {'trend': 'stable', 'trend_strength': 0.0}
    
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

**Uso no Backend**:
```typescript
if (semantic_trend.trend === 'advancing' && 
    semantic_trend.current_stage >= 3) {
  // Cliente progredindo para estágios avançados
  // Gerar feedback positivo
}
```

### 4.4 Redução de Ruído de Frases Isoladas

**Problema**: Frase isolada pode gerar classificação incorreta

**Solução**: Requerer consistência em janela temporal

**Estratégia**:
1. Classificar chunk atual
2. Verificar se categoria é consistente com histórico
3. Se inconsistente, usar categoria agregada do histórico
4. Se histórico insuficiente, usar chunk atual mas marcar como "low_confidence"

**Implementação**:
```python
def classify_with_context(
    self,
    text: str,
    context_window: List[Dict[str, Any]],
    min_consistency: float = 0.6
) -> Dict[str, Any]:
    """
    Classifica texto considerando contexto histórico.
    
    Args:
        text: Texto atual
        context_window: Janela de contexto histórico
        min_consistency: Consistência mínima para aceitar categoria atual
    
    Returns:
        {
            'category': str,
            'confidence': float,
            'is_consistent': bool,
            'used_context': bool  # True se usou contexto ao invés de chunk atual
        }
    """
    # Classificar chunk atual
    current_cat, current_conf, scores = self.classify_sales_category(text)
    
    if not context_window:
        return {
            'category': current_cat,
            'confidence': current_conf,
            'is_consistent': True,
            'used_context': False
        }
    
    # Agregar categorias do histórico
    aggregated = self.aggregate_categories_temporal(context_window)
    dominant_historical = aggregated['dominant_category']
    
    # Verificar consistência
    is_consistent = (
        current_cat == dominant_historical or
        aggregated['stability'] < 0.5  # Histórico instável, aceitar atual
    )
    
    if is_consistent or current_conf > 0.8:
        # Usar categoria atual
        return {
            'category': current_cat,
            'confidence': current_conf,
            'is_consistent': is_consistent,
            'used_context': False
        }
    else:
        # Usar categoria histórica (mais confiável)
        return {
            'category': dominant_historical,
            'confidence': aggregated['stability'],
            'is_consistent': False,
            'used_context': True
        }
```

---

## 5️⃣ Sinais Semânticos Padronizados (Contrato Python → Backend)

### 5.1 Estrutura de Saída Expandida

**Situação Atual**:
```json
{
  "sales_category": "price_interest",
  "sales_category_confidence": 0.85
}
```

**Estrutura Proposta**:
```json
{
  "semantic_signals": {
    "sales_category": {
      "primary": "price_interest",
      "secondary": ["information_gathering"],  // Multi-label quando aplicável
      "confidence": 0.85,
      "intensity": 0.92,  // Score absoluto da melhor categoria
      "ambiguity": 0.15,   // Quão ambíguo é o texto (0=claro, 1=muito ambíguo)
      "scores": {
        "price_interest": 0.92,
        "information_gathering": 0.68,
        "value_exploration": 0.45,
        ...
      }
    },
    "context": {
      "trend": "advancing",  // advancing | stable | regressing
      "trend_strength": 0.75,
      "current_stage": 3,    // Posição na progressão (1-5)
      "stability": 0.82,     // Estabilidade da categoria na janela
      "consistency": true    // Se categoria atual é consistente com histórico
    },
    "transitions": {
      "detected": true,
      "from_category": "value_exploration",
      "to_category": "price_interest",
      "transition_type": "advancing",
      "confidence": 0.88,
      "time_delta_ms": 15000
    },
    "flags": {
      "price_window_open": true,        // Janela de oportunidade para preço
      "decision_signal_strong": false,   // Sinal forte de decisão
      "objection_escalating": false,    // Objeção piorando
      "conversation_stalling": false,    // Conversa estagnada
      "ready_to_close": false           // Pronto para fechar
    }
  }
}
```

### 5.2 Flags Semânticas Específicas

**Proposta**: Flags booleanas que facilitam heurísticas no backend

**Flags Propostas**:

```python
def generate_semantic_flags(
    self,
    category: str,
    confidence: float,
    intensity: float,
    context: Dict[str, Any],
    transitions: Optional[Dict[str, Any]]
) -> Dict[str, bool]:
    """
    Gera flags semânticas baseadas em análise completa.
    
    Flags são booleanas e facilitam decisões no backend.
    """
    flags = {}
    
    # Flag: Janela de oportunidade para preço
    flags['price_window_open'] = (
        category == 'price_interest' and
        confidence > 0.7 and
        intensity > 0.8
    )
    
    # Flag: Sinal forte de decisão
    flags['decision_signal_strong'] = (
        category in ['decision_signal', 'closing_readiness'] and
        confidence > 0.8 and
        intensity > 0.85
    )
    
    # Flag: Objeção escalando
    flags['objection_escalating'] = (
        transitions and
        transitions['from_category'] == 'objection_soft' and
        transitions['to_category'] == 'objection_hard' and
        transitions['transition_type'] == 'regressing'
    )
    
    # Flag: Conversa estagnada
    flags['conversation_stalling'] = (
        context.get('trend') == 'stable' and
        context.get('stability', 0) > 0.9 and
        category == 'stalling'
    )
    
    # Flag: Pronto para fechar
    flags['ready_to_close'] = (
        category == 'closing_readiness' and
        confidence > 0.85 and
        context.get('trend') == 'advancing' and
        context.get('current_stage', 0) >= 4
    )
    
    return flags
```

**Uso no Backend**:
```typescript
if (semantic_signals.flags.price_window_open) {
  // Gerar feedback: "Agora é o momento de falar sobre preço"
}

if (semantic_signals.flags.objection_escalating) {
  // Gerar feedback urgente sobre objeção
}

if (semantic_signals.flags.ready_to_close) {
  // Gerar feedback: "Cliente pronto para fechar - acelerar!"
}
```

### 5.3 Intensidade do Sinal

**Proposta**: Score absoluto da melhor categoria (diferente de confiança)

**Diferença**:
- **Confiança**: Diferença relativa entre melhor e segunda melhor (0-1)
- **Intensidade**: Score absoluto da melhor categoria (0-1)

**Exemplo**:
```
Caso 1:
  Melhor: 0.9, Segunda: 0.2
  Confiança: (0.9-0.2)/0.9 = 0.78 (alta)
  Intensidade: 0.9 (alta)

Caso 2:
  Melhor: 0.5, Segunda: 0.1
  Confiança: (0.5-0.1)/0.5 = 0.8 (alta)
  Intensidade: 0.5 (média)
```

**Uso**: Backend pode usar intensidade para priorizar feedbacks

### 5.4 Direção da Conversa

**Proposta**: Indicador de progressão/regressão

**Valores**:
- `advancing`: Cliente progredindo (ex: value → price → decision)
- `stable`: Sem mudança significativa
- `regressing`: Cliente regredindo (ex: decision → objection)

**Cálculo**: Baseado em tendência semântica (seção 4.3)

**Uso no Backend**:
```typescript
if (semantic_signals.context.trend === 'regressing') {
  // Gerar alerta: "Cliente regredindo - requer atenção"
}
```

---

## 6️⃣ Heurísticas no Backend (Node/NestJS)

### 6.1 Combinação de Sinais Semânticos + Tempo + Histórico

**Estratégia**: Backend combina múltiplos sinais para gerar feedback confiável

**Heurística Proposta**:
```typescript
function shouldGenerateSalesFeedback(
  state: ParticipantState,
  semanticSignals: SemanticSignals,
  now: number
): boolean {
  // 1. Verificar cooldown global
  if (inGlobalCooldown(state, now, 30000)) { // 30s
    return false;
  }
  
  // 2. Verificar flags semânticas fortes
  if (semanticSignals.flags.decision_signal_strong ||
      semanticSignals.flags.objection_escalating ||
      semanticSignals.flags.ready_to_close) {
    return true; // Flags fortes sempre geram feedback
  }
  
  // 3. Verificar consistência temporal
  const recentCategories = getRecentCategories(state, 60000); // Último minuto
  const consistency = calculateConsistency(
    semanticSignals.sales_category.primary,
    recentCategories
  );
  
  if (consistency < 0.6 && semanticSignals.context.consistency === false) {
    return false; // Muito inconsistente, não gerar feedback
  }
  
  // 4. Verificar confiança e intensidade
  if (semanticSignals.sales_category.confidence < 0.6 ||
      semanticSignals.sales_category.intensity < 0.6) {
    return false; // Muito incerto
  }
  
  // 5. Verificar ambiguidade
  if (semanticSignals.sales_category.ambiguity > 0.7) {
    return false; // Muito ambíguo
  }
  
  return true;
}
```

### 6.2 Evitar Feedbacks Prematuros

**Problema**: Feedback gerado muito cedo pode ser baseado em ruído

**Solução**: Requerer estabilidade temporal

**Heurística**:
```typescript
function isFeedbackPremature(
  state: ParticipantState,
  semanticSignals: SemanticSignals,
  now: number
): boolean {
  // Requerer pelo menos 2 chunks com mesma categoria
  const recentCategories = getRecentCategories(state, 30000); // 30s
  const sameCategoryCount = recentCategories.filter(
    cat => cat === semanticSignals.sales_category.primary
  ).length;
  
  if (sameCategoryCount < 2 && !semanticSignals.flags.decision_signal_strong) {
    return true; // Muito prematuro
  }
  
  // Requerer estabilidade mínima
  if (semanticSignals.context.stability < 0.5) {
    return true; // Muito instável
  }
  
  return false;
}
```

### 6.3 Gerar Feedbacks Acionáveis e Contextualizados

**Estratégia**: Mensagens específicas baseadas em combinação de sinais

**Exemplos de Heurísticas**:

```typescript
function generateSalesFeedback(
  state: ParticipantState,
  semanticSignals: SemanticSignals
): FeedbackEventPayload | null {
  // Heurística 1: Janela de preço
  if (semanticSignals.flags.price_window_open &&
      semanticSignals.context.trend === 'advancing') {
    return {
      type: 'sales_opportunity',
      severity: 'info',
      message: 'Agora é o momento ideal para apresentar o preço',
      tips: [
        'Cliente demonstrou interesse consistente',
        'Conversa progredindo positivamente',
        'Confiança alta na classificação'
      ]
    };
  }
  
  // Heurística 2: Objeção escalando
  if (semanticSignals.flags.objection_escalating) {
    return {
      type: 'sales_alert',
      severity: 'warning',
      message: 'Objeção do cliente está piorando - requer abordagem diferente',
      tips: [
        'Cliente regrediu de objeção leve para forte',
        'Considerar mudança de estratégia',
        'Focar em entender preocupações específicas'
      ]
    };
  }
  
  // Heurística 3: Pronto para fechar
  if (semanticSignals.flags.ready_to_close &&
      semanticSignals.context.current_stage >= 4) {
    return {
      type: 'sales_opportunity',
      severity: 'info',
      message: 'Cliente demonstra prontidão para fechar - acelerar processo',
      tips: [
        'Múltiplos sinais de fechamento detectados',
        'Conversa progredindo consistentemente',
        'Momento ideal para proposta final'
      ]
    };
  }
  
  // Heurística 4: Conversa estagnada
  if (semanticSignals.flags.conversation_stalling &&
      semanticSignals.context.trend === 'stable') {
    return {
      type: 'sales_alert',
      severity: 'info',
      message: 'Conversa estagnada - considerar criar urgência',
      tips: [
        'Cliente protelando decisão',
        'Considerar oferecer incentivo ou deadline',
        'Revisar valor proposto'
      ]
    };
  }
  
  return null;
}
```

### 6.4 Priorização de Feedbacks

**Problema**: Múltiplos feedbacks podem ser gerados simultaneamente

**Solução**: Sistema de prioridades

**Prioridades**:
1. **Crítica**: `objection_escalating`, `ready_to_close`
2. **Alta**: `price_window_open`, `decision_signal_strong`
3. **Média**: `conversation_stalling`, transições importantes
4. **Baixa**: Categorias estáveis sem flags

**Implementação**:
```typescript
const FEEDBACK_PRIORITIES = {
  'objection_escalating': 10,
  'ready_to_close': 10,
  'price_window_open': 8,
  'decision_signal_strong': 8,
  'conversation_stalling': 5,
  'default': 3
};

function prioritizeFeedback(feedback: FeedbackEventPayload): number {
  // Extrair tipo do feedback
  const type = feedback.type;
  return FEEDBACK_PRIORITIES[type] || FEEDBACK_PRIORITIES.default;
}

// No aggregator:
const feedbacks = [
  generateSalesFeedback(state, signals),
  generateEmotionalFeedback(state, ctx),
  // ... outros feedbacks
].filter(f => f !== null);

if (feedbacks.length > 0) {
  // Selecionar feedback de maior prioridade
  const topFeedback = feedbacks.reduce((a, b) => 
    prioritizeFeedback(a) > prioritizeFeedback(b) ? a : b
  );
  
  this.delivery.publishToHosts(meetingId, topFeedback);
}
```

---

## 7️⃣ Performance e Escalabilidade

### 7.1 Cache de Embeddings

**Situação Atual**: ✅ Cache de exemplos implementado

**Melhorias Propostas**:

**Cache de Embeddings de Textos**:
```python
class EmbeddingCache:
    """
    Cache de embeddings de textos para evitar recálculo.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Retorna embedding se em cache e válido"""
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            embedding, timestamp = self.cache[text_hash]
            if time.time() - timestamp < self.ttl_seconds:
                return embedding
            else:
                del self.cache[text_hash]
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Armazena embedding no cache"""
        text_hash = self._hash_text(text)
        if len(self.cache) >= self.max_size:
            # Remover mais antigo (LRU simplificado)
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]
        
        self.cache[text_hash] = (embedding, time.time())
```

**Uso**: Cachear embeddings de chunks para comparações históricas

### 7.2 Batch Processing

**Problema**: Processar múltiplos textos sequencialmente é lento

**Solução**: Processar em batch quando possível

**Implementação**:
```python
def classify_sales_category_batch(
    self,
    texts: List[str],
    min_confidence: float = 0.3
) -> List[Tuple[Optional[str], float, Dict[str, float]]]:
    """
    Classifica múltiplos textos em batch (mais eficiente).
    
    Performance: ~2x mais rápido que processar sequencialmente
    """
    if not self._sales_examples_loaded:
        self._load_sales_category_examples_embeddings()
    
    # Gerar embeddings em batch
    text_embeddings = self.sbert_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    )
    
    results = []
    for text_embedding in text_embeddings:
        # ... cálculo de similaridade ...
        results.append((category, confidence, scores))
    
    return results
```

**Uso**: Quando processar histórico ou múltiplos chunks

### 7.3 Limites de Latência Aceitáveis

**Metas de Performance**:

| Operação | Latência Aceitável | Latência Ideal |
|----------|-------------------|----------------|
| Primeira análise (com contexto) | < 1.5s | < 1.0s |
| Análises subsequentes | < 100ms | < 50ms |
| Classificação batch (10 textos) | < 500ms | < 300ms |
| Cálculo de tendência | < 50ms | < 20ms |

**Estratégias**:
- Cache agressivo
- Processamento assíncrono quando possível
- Limitar tamanho de janelas de contexto
- Pré-calcular métricas quando possível

### 7.4 Estratégia CPU-First com GPU Opcional

**Situação Atual**: CPU-only

**Estratégia**:
1. **Curto Prazo**: Otimizar para CPU
   - Usar modelos leves
   - Cache agressivo
   - Batch processing quando possível

2. **Médio Prazo**: Suporte opcional a GPU
   - Detectar GPU automaticamente
   - Usar GPU se disponível
   - Fallback para CPU

3. **Longo Prazo**: Avaliar necessidade de GPU
   - Se latência CPU for aceitável, manter CPU
   - Se necessário, considerar GPU dedicada

**Implementação**:
```python
# Já implementado: device detection
if self.device == "cuda" and torch.cuda.is_available():
    self.model = self.model.to("cuda")
```

### 7.5 Separação Clara: Transcrição vs Análise Semântica

**Arquitetura Atual**: ✅ Já separado

**Melhorias**:
- Garantir que falha em análise semântica não bloqueia transcrição
- Processar análise semântica de forma assíncrona quando possível
- Priorizar transcrição sobre análise (transcrição é crítica)

---

## 8️⃣ Observabilidade e Qualidade

### 8.1 Métricas Semânticas

**Métricas Propostas**:

```python
class SemanticMetrics:
    """
    Coleta métricas de qualidade da classificação semântica.
    """
    def __init__(self):
        self.metrics = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'avg_confidence': 0.0,
            'avg_intensity': 0.0,
            'avg_ambiguity': 0.0,
            'category_distribution': {},
            'transition_count': 0,
            'high_confidence_rate': 0.0  # % com confiança > 0.7
        }
    
    def record_classification(
        self,
        category: Optional[str],
        confidence: float,
        intensity: float,
        ambiguity: float
    ):
        """Registra uma classificação"""
        self.metrics['total_classifications'] += 1
        
        if category:
            self.metrics['successful_classifications'] += 1
            self.metrics['category_distribution'][category] = \
                self.metrics['category_distribution'].get(category, 0) + 1
        else:
            self.metrics['failed_classifications'] += 1
        
        # Atualizar médias (média móvel exponencial)
        alpha = 0.1
        self.metrics['avg_confidence'] = (
            alpha * confidence + (1 - alpha) * self.metrics['avg_confidence']
        )
        self.metrics['avg_intensity'] = (
            alpha * intensity + (1 - alpha) * self.metrics['avg_intensity']
        )
        self.metrics['avg_ambiguity'] = (
            alpha * ambiguity + (1 - alpha) * self.metrics['avg_ambiguity']
        )
        
        if confidence > 0.7:
            self.metrics['high_confidence_rate'] = (
                self.metrics['high_confidence_rate'] * 0.99 + 0.01
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais"""
        return self.metrics.copy()
```

**Exposição**: Endpoint `/metrics` no FastAPI

### 8.2 Logs Explicáveis

**Problema**: Logs não explicam por que um sinal foi emitido

**Solução**: Logs estruturados com contexto completo

**Formato Proposto**:
```python
logger.info(
    "Semantic signal generated",
    signal_type="price_window_open",
    reasoning={
        "category": "price_interest",
        "confidence": 0.85,
        "intensity": 0.92,
        "ambiguity": 0.15,
        "context_consistency": True,
        "temporal_stability": 0.82,
        "trend": "advancing",
        "flags_triggered": ["price_window_open"],
        "why": "High confidence price_interest with advancing trend and low ambiguity"
    }
)
```

**Uso**: Facilita debugging e validação manual

### 8.3 Estratégia de Validação Manual

**Proposta**: Dashboard de validação

**Funcionalidades**:
1. Visualizar classificações em tempo real
2. Marcar classificações como corretas/incorretas
3. Ver métricas de qualidade
4. Ajustar thresholds baseado em feedback

**Implementação Futura**:
- Endpoint `/validate` para marcar classificações
- Armazenar validações para análise
- Ajustar exemplos baseado em feedback

### 8.4 Ajustes Contínuos via Dados Reais

**Estratégia**: Coletar dados e melhorar iterativamente

**Coleta de Dados**:
1. Logs de classificações (anônimos)
2. Validações manuais quando possível
3. Métricas de uso (quais categorias mais comuns)

**Melhorias Baseadas em Dados**:
1. Ajustar exemplos de referência
2. Ajustar thresholds de confiança
3. Adicionar novas categorias se necessário
4. Remover categorias pouco usadas

---

## 9️⃣ Roadmap Incremental

### Curto Prazo (1-2 semanas) - Quick Wins

**Objetivo**: Melhorias rápidas com alto impacto

**Itens**:

1. **Expandir Exemplos de Referência**
   - Adicionar 5 exemplos por categoria (80 → 120 total)
   - Incluir variações regionais
   - **Esforço**: 4 horas
   - **Impacto**: +10-15% precisão

2. **Adicionar Score de Ambiguidade**
   - Implementar cálculo de ambiguidade
   - Incluir no retorno
   - **Esforço**: 2 horas
   - **Impacto**: Reduz falsos positivos

3. **Adicionar Intensidade do Sinal**
   - Score absoluto da melhor categoria
   - Incluir no retorno
   - **Esforço**: 1 hora
   - **Impacto**: Backend pode priorizar melhor

4. **Melhorar Logging**
   - Logs mais explicáveis
   - Incluir reasoning
   - **Esforço**: 2 horas
   - **Impacto**: Melhor debugging

5. **Flags Semânticas Básicas**
   - Implementar 3-5 flags principais
   - `price_window_open`, `decision_signal_strong`, `ready_to_close`
   - **Esforço**: 4 horas
   - **Impacto**: Backend pode gerar feedbacks mais específicos

**Total**: ~13 horas de desenvolvimento

### Médio Prazo (1-2 meses)

**Objetivo**: Funcionalidades mais complexas

**Itens**:

1. **Análise de Contexto Conversacional**
   - Implementar `ConversationContext`
   - Janelas temporais
   - Agregação temporal
   - **Esforço**: 16 horas
   - **Impacto**: Reduz ruído, detecta padrões

2. **Detecção de Transições**
   - Comparar categoria atual com histórico
   - Detectar progressão/regressão
   - **Esforço**: 8 horas
   - **Impacto**: Detecta mudanças importantes

3. **Tendência Semântica**
   - Calcular tendência ao longo do tempo
   - Direção da conversa
   - **Esforço**: 6 horas
   - **Impacto**: Backend pode gerar feedbacks contextuais

4. **Classificação Multi-Label**
   - Permitir múltiplas categorias
   - **Esforço**: 8 horas
   - **Impacto**: Captura casos complexos

5. **Heurísticas no Backend**
   - Implementar `shouldGenerateSalesFeedback`
   - Priorização de feedbacks
   - **Esforço**: 12 horas
   - **Impacto**: Feedbacks mais confiáveis

6. **Métricas e Observabilidade**
   - Coletar métricas semânticas
   - Endpoint `/metrics`
   - **Esforço**: 6 horas
   - **Impacto**: Visibilidade de qualidade

**Total**: ~56 horas de desenvolvimento

### Longo Prazo (3-6 meses)

**Objetivo**: Otimizações e melhorias avançadas

**Itens**:

1. **Fine-tuning do Modelo SBERT**
   - Coletar dataset de reuniões reais
   - Anotar manualmente
   - Fine-tune para domínio específico
   - **Esforço**: 40+ horas
   - **Impacto**: +20-30% precisão

2. **Cache de Embeddings de Textos**
   - Implementar cache LRU
   - Reduzir recálculos
   - **Esforço**: 4 horas
   - **Impacto**: Melhor performance

3. **Batch Processing**
   - Processar múltiplos textos em batch
   - **Esforço**: 6 horas
   - **Impacto**: 2x mais rápido para histórico

4. **Dashboard de Validação**
   - Interface para validar classificações
   - Coletar feedback
   - **Esforço**: 20 horas
   - **Impacto**: Melhorar qualidade iterativamente

5. **Ajustes Baseados em Dados**
   - Analisar dados coletados
   - Ajustar exemplos e thresholds
   - **Esforço**: Contínuo
   - **Impacto**: Melhoria contínua

**Total**: ~70+ horas de desenvolvimento

### O Que NÃO Fazer Agora (Anti-Overengineering)

**Evitar**:
- ❌ Modelos muito complexos (ex: LLMs grandes)
- ❌ Fine-tuning sem dados reais suficientes
- ❌ GPU dedicada antes de otimizar CPU
- ❌ Sistema de ML completo (manter simples)
- ❌ Over-engineering de cache (atual é suficiente)
- ❌ Muitas categorias novas sem validação

**Princípio**: Implementar apenas o necessário, validar com dados reais, iterar.

---

## 🔟 Decisões Técnicas Principais

### Decisão 1: Manter SBERT Atual vs Fine-tuning

**Decisão**: Manter modelo atual por agora, avaliar fine-tuning após coletar dados

**Razão**: Fine-tuning requer dataset anotado, que ainda não temos. Melhor validar abordagem atual primeiro.

### Decisão 2: Contexto em Python vs Backend

**Decisão**: Contexto em Python (janelas temporais), histórico completo no backend

**Razão**: Python já tem embeddings, mais eficiente calcular similaridades lá. Backend mantém histórico completo para heurísticas complexas.

### Decisão 3: Multi-Label vs Single-Label

**Decisão**: Implementar multi-label opcional (médio prazo)

**Razão**: Útil para casos complexos, mas não crítico. Pode adicionar depois.

### Decisão 4: Flags vs Scores Diretos

**Decisão**: Ambos - flags para facilitar heurísticas, scores para flexibilidade

**Razão**: Flags facilitam código no backend, scores permitem heurísticas customizadas.

### Decisão 5: CPU-First vs GPU-First

**Decisão**: CPU-first com suporte opcional a GPU

**Razão**: CPU é suficiente para latência aceitável, GPU adiciona complexidade e custo.

---

## 📊 Métricas de Sucesso

### Métricas Técnicas

- **Precisão de Classificação**: > 80% (validado manualmente)
- **Latência P95**: < 100ms (análises subsequentes)
- **Taxa de Falsos Positivos**: < 10%
- **Cobertura de Categorias**: Todas as 8 categorias detectáveis

### Métricas de Negócio

- **Feedbacks Gerados**: Taxa adequada (não muito, não pouco)
- **Qualidade dos Feedbacks**: Validado por vendedores
- **Ação dos Vendedores**: Feedbacks levam a ações

### Métricas de Qualidade

- **Confiança Média**: > 0.7
- **Ambiguidade Média**: < 0.4
- **Taxa de Alta Confiança**: > 60%

---

## 🎯 Conclusão

Este roadmap fornece um plano incremental e pragmático para evoluir o sistema de análise semântica de vendas. As melhorias são projetadas para:

1. **Aumentar Robustez**: Contexto e agregação temporal
2. **Melhorar Precisão**: Mais exemplos, métricas de qualidade
3. **Facilitar Heurísticas**: Flags e sinais estruturados
4. **Manter Simplicidade**: Incremental, sem over-engineering

**Próximo Passo Recomendado**: Implementar quick wins (curto prazo) e validar com dados reais antes de avançar para melhorias mais complexas.

---

**Documento criado em**: 2025-01-XX  
**Versão**: 1.0  
**Status**: Proposta Técnica

