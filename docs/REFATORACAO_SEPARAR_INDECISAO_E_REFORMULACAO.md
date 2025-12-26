# Refatoração: Separar Análises de Indecisão e Reformulação

## 📋 Índice

1. [Objetivo e Escopo](#objetivo-e-escopo)
2. [Estado Atual](#estado-atual)
3. [Arquitetura Alvo](#arquitetura-alvo)
4. [Passos de Execução](#passos-de-execução)
   - [Passo 0: Preparação](#passo-0-preparação)
   - [Passo 1: Criar Estrutura](#passo-1-criar-estrutura)
   - [Passo 2: Extrair Reformulação](#passo-2-extrair-reformulação)
   - [Passo 3: Extrair Indecisão](#passo-3-extrair-indecisão)
   - [Passo 4: Integrar no Analysis Service](#passo-4-integrar-no-analysis-service)
   - [Passo 5: Limpeza](#passo-5-limpeza)
   - [Passo 6: Testes](#passo-6-testes)
   - [Passo 7: Verificação Manual](#passo-7-verificação-manual)
5. [Checklist Final](#checklist-final)
6. [Observações Importantes](#observações-importantes)

---

## Objetivo e Escopo

### Objetivo (o que você vai ganhar)

Separar, em módulos próprios, tudo que hoje é calculado dentro de `TextAnalysisService.analyze()` e que corresponde a:

- **Indecisão**: cálculo de `indecision_metrics` (baseado em `sales_category_*` + `conditional_keywords_detected`)
- **Reformulação (“solução foi compreendida”)**: detecção de `reformulation_markers_detected`, cálculo de `reformulation_marker_score` e **efeito colateral** no `sales_category_flags['solution_reformulation_signal']`

Mantendo **100% a mesma lógica** (mesmas condições, mesmos thresholds, mesmos valores de retorno, mesmas chaves no payload), mas deixando cada “análise” isolada em seu arquivo, de modo que você consiga:

- editar **Indecisão** sem encostar no código de **Reformulação**
- editar **Reformulação** sem encostar no código de **Indecisão**
- reduzir o tamanho/complexidade de `analysis_service.py` e facilitar review

---

### Escopo e restrições (o que NÃO pode mudar)

Este plano é uma **refatoração estrutural** (“move code”), portanto:

- **Não alterar nomes de chaves** no resultado final:
  - `analysis.indecision_metrics`
  - `analysis.reformulation_markers_detected`
  - `analysis.reformulation_marker_score`
  - `analysis.sales_category_flags.solution_reformulation_signal`
- **Não alterar lógica e ordem** de execução:
  - Reformulação e Indecisão continuam sendo calculadas **no mesmo ponto do fluxo** (após classificação SBERT; antes de `record_classification`).
  - A flag `solution_reformulation_signal` continua sendo aplicada **antes** de `self.metrics.record_classification(...)`, para que a métrica de flags continue contando do mesmo jeito.
- **Não alterar regras de gating**:
  - Reformulação continua rodando sempre (depende apenas de `chunk.text`).
  - Indecisão continua rodando **somente se**:
    - `Config.SBERT_MODEL_NAME` é truthy **E**
    - `sales_category is not None`
- **Não alterar tratamento de erro**:
  - Se qualquer parte falhar, continua “engolindo” e seguindo o fluxo (sem quebrar a análise completa).
- **Não alterar o cálculo**:
  - `reformulation_marker_score = min(1.0, len(markers) / 2.0)`
  - `solution_reformulation_signal` só é setado quando `reformulation_marker_score > 0.0`
  - `calculate_indecision_metrics` continua igual (permanece em `BERTAnalyzer.calculate_indecision_metrics(...)` ou é movido sem mudanças literais)
- **Não alterar o cache** (`AnalysisCache`) e nem a chave do cache.

---

## Estado Atual

### Mapa Exato do que Separar

#### Onde a separação hoje “está misturada”

Arquivo: `apps/text-analysis/src/services/analysis_service.py`

Dentro de `class TextAnalysisService`, método `async def analyze(self, chunk: TranscriptionChunk)`.

Após a classificação SBERT (bloco “CLASSIFICAÇÃO DE CATEGORIAS DE VENDAS COM SBERT”), existem dois blocos consecutivos:

1) **Reformulação (teach-back / “solução foi compreendida”)**:

```python
# (Opcional) Reformulação do cliente ("solução foi compreendida")
reformulation_markers_detected = self._detect_reformulation_markers(chunk.text)
reformulation_marker_score = min(1.0, len(reformulation_markers_detected) / 2.0)
if reformulation_marker_score > 0.0:
    # Flag genérica para heurísticas no backend (não depende de category)
    sales_category_flags['solution_reformulation_signal'] = True
```

2) **Indecisão (métricas)**:

```python
# FASE 10: CÁLCULO DE MÉTRICAS DE INDECISÃO
indecision_metrics: Dict[str, Any] = {}
try:
    if Config.SBERT_MODEL_NAME and sales_category is not None:
        indecision_metrics = analyzer.calculate_indecision_metrics(
            sales_category,
            sales_category_confidence or 0.0,
            sales_category_intensity or 0.0,
            sales_category_ambiguity or 0.0,
            conditional_keywords_detected
        )
except Exception as e:
    logger.warn(...)
```

E, no final, os campos são colocados no `result`:

```python
result = {
  # ...
  'sales_category_flags': sales_category_flags,
  'conditional_keywords_detected': conditional_keywords_detected,
  'indecision_metrics': indecision_metrics if indecision_metrics else None,
  'reformulation_markers_detected': reformulation_markers_detected,
  'reformulation_marker_score': reformulation_marker_score
}
```

#### Onde está a lógica “core” de indecisão (já bem isolada)

Arquivo: `apps/text-analysis/src/models/bert_analyzer.py`

- `def calculate_indecision_metrics(...):` contém a fórmula e regras de cálculo:
  - `indecision_score`
  - `postponement_likelihood`
  - `conditional_language_score`

Ou seja: **o algoritmo em si já está isolado**. O que está “misturado” em `analysis_service.py` é:

- a regra **quando calcular** (gating)
- o **try/except** e o fallback `{}`
- o local onde isso é acoplado no payload

#### Onde está a lógica “core” de reformulação (hoje dentro do serviço)

Arquivo: `apps/text-analysis/src/services/analysis_service.py`

`def _detect_reformulation_markers(self, text: str) -> List[str]:` contém:

- lista de marcadores (PT-BR)
- regra de matching: `if m in t` (substring) com `t = (text or "").lower()`
- ordem do retorno: **na ordem do array `markers`**

---

## Arquitetura Alvo

### Como Deve Ficar Após a Refatoração

### Regra principal

Cada “análise” vira um **módulo independente** em uma pasta nova `src/signals/`, com:

- **reformulation.py**: só trata reformulação
- **indecision.py**: só trata indecisão

O `analysis_service.py` continua sendo o **orquestrador**, mas ele só chama funções bem nomeadas, sem carregar a lógica inline.

### Estrutura de pastas proposta

Dentro de `apps/text-analysis/src/`:

```text
src/
  signals/
    __init__.py
    reformulation.py
    indecision.py
```

### Interfaces alvo (assinaturas e responsabilidades)

#### `src/signals/reformulation.py`

Responsabilidade: dado um texto, produzir:

- `reformulation_markers_detected: List[str]`
- `reformulation_marker_score: float`
- e aplicar o **mesmo efeito colateral** no `sales_category_flags` (setar `solution_reformulation_signal=True` quando score > 0.0)

Interface sugerida (mantendo comportamento idêntico):

- `detect_reformulation_markers(text: str) -> List[str]`
  - **deve ser uma cópia literal** da lógica de `_detect_reformulation_markers`
  - manter strings e ordem
  - manter `(text or "").lower()`
  - manter substring `if m in t`

- `compute_reformulation_marker_score(markers: List[str]) -> float`
  - **deve ser literal**: `min(1.0, len(markers) / 2.0)`

- `apply_solution_reformulation_signal_flag(flags: Dict[str, bool], marker_score: float) -> None`
  - deve fazer exatamente o que hoje ocorre:
    - se `marker_score > 0.0`: `flags['solution_reformulation_signal'] = True`
    - caso contrário: **não mexer no dict** (não criar key, não setar False)

Observação importante: essa “aplicação de flag” é o ponto que liga Reformulação ao bloco de vendas. Separar em função evita que alguém esqueça o side-effect.

#### `src/signals/indecision.py`

Responsabilidade: encapsular o “bloco de orquestração” de indecisão que hoje está em `analysis_service.py`, sem mexer no algoritmo dentro do `BERTAnalyzer`.

Interface sugerida:

- `compute_indecision_metrics_safe(...) -> Dict[str, Any]`
  - retorna `{}` quando não calcula (mesmo fallback atual)
  - faz o mesmo gating:
    - SBERT habilitado
    - `sales_category is not None`
  - chama `analyzer.calculate_indecision_metrics(...)` com os mesmos parâmetros e fallback `or 0.0`
  - tem o mesmo try/except (não quebra o fluxo)

Entrada recomendada (para manter lógica idêntica):

- `analyzer: BERTAnalyzer`
- `sbert_enabled: bool` (derivado de `bool(Config.SBERT_MODEL_NAME)`)
- `sales_category: Optional[str]`
- `sales_category_confidence: Optional[float]`
- `sales_category_intensity: Optional[float]`
- `sales_category_ambiguity: Optional[float]`
- `conditional_keywords_detected: List[str]`
- `meeting_id: str` (apenas para logs, se você quiser preservar logs)

Saída:

- `Dict[str, Any]` (ex.: `{'indecision_score': 0.8, ...}`) ou `{}` se não calculou / falhou

---

## Passos de Execução

### Passo 0: Preparação

**Objetivo:** Garantir que qualquer mudança estrutural não altere output.

**Tarefas:**

1. **Criar branch/PR dedicada:**
   ```bash
   git checkout -b refactor/separate-indecision-reformulation
   ```

2. **Rodar testes existentes:**
   ```bash
   cd apps/text-analysis
   pytest -q
   ```
   - Verificar que todos passam antes de começar

3. **Criar snapshot de outputs (golden outputs):**
   - Escolher conjunto pequeno de textos de teste:
     - Com marcadores: `"Deixa eu ver se entendi, então vocês fazem X e Y?"`
     - Sem marcadores: `"Ok, entendi"`
     - Com `sales_category` setado (mock) e sem (mock)
   - **Ideal:** Criar teste que:
     - Chama `TextAnalysisService.analyze(...)` com analyzer mockado
     - Compara o `analysis` resultante com dict esperado
     - Salva como "snapshot" para comparação futura

**Por que isso é obrigatório?**
Mover código tende a:
- Mudar condições (ex.: `if score` vs `if score > 0.0`)
- Mudar defaults (`or 0.0` vs `if is None`)
- Mudar ordem de efeitos colaterais (flag antes/depois da métrica)

O snapshot evita regressão invisível.

---

### Passo 1: Criar Estrutura

**Objetivo:** Criar a estrutura de pastas para os novos módulos.

**Tarefas:**

1. Criar diretório:
   ```bash
   mkdir -p apps/text-analysis/src/signals
   ```

2. Criar arquivo `__init__.py`:
   ```bash
   touch apps/text-analysis/src/signals/__init__.py
   ```
   - Pode estar vazio; serve para tornar o pacote importável

---

### Passo 2: Extrair Reformulação

**Objetivo:** Mover **toda** a lógica específica de reformulação para um arquivo só.

**Arquivo alvo:** `apps/text-analysis/src/signals/reformulation.py`

**Tarefas:**

1. **Criar arquivo:**
   ```bash
   touch apps/text-analysis/src/signals/reformulation.py
   ```

2. **Copiar método `_detect_reformulation_markers`:**
   - Localizar em: `apps/text-analysis/src/services/analysis_service.py` (linha ~737)
   - Copiar **literalmente** (sem alterações) para `reformulation.py`
   - Renomear para função pública: `def detect_reformulation_markers(text: str) -> List[str]`
   - Remover `self` (não é mais método de classe)

   **Checklist de paridade (não negocie):**
   - [ ] Mantém `t = (text or "").lower()`
   - [ ] Mantém `if m in t` (substring, não regex)
   - [ ] Mantém ordem do retorno (na ordem do array `markers`)
   - [ ] Mantém as mesmas strings e acentos:
     - `"só pra confirmar"` com acento, etc.

3. **Criar função de score:**
   ```python
   def compute_reformulation_marker_score(markers: List[str]) -> float:
       return min(1.0, len(markers) / 2.0)
   ```
   - Deve ser **exatamente** como acima (literal)

4. **Criar função de side-effect (flag):**
   ```python
   def apply_solution_reformulation_signal_flag(
       flags: Dict[str, bool],
       marker_score: float
   ) -> None:
       if marker_score > 0.0:
           flags['solution_reformulation_signal'] = True
   ```
   - **Não** criar key se `score <= 0.0` (não setar `False`, não criar key)

5. **Adicionar imports necessários:**
   ```python
   from typing import List, Dict
   ```

**Por que fazer a flag numa função?**
Hoje essa flag é um detalhe fácil de esquecer; separar aumenta segurança de manutenção:
- Você altera o detector sem precisar lembrar de setar a flag em outro arquivo

---

### Passo 3: Extrair Indecisão

**Objetivo:** Tirar de `analysis_service.py` o bloco "FASE 10" (try/except + gating + call).

**Arquivo alvo:** `apps/text-analysis/src/signals/indecision.py`

**Tarefas:**

1. **Criar arquivo:**
   ```bash
   touch apps/text-analysis/src/signals/indecision.py
   ```

2. **Implementar função `compute_indecision_metrics_safe`:**
   - Localizar bloco em: `apps/text-analysis/src/services/analysis_service.py` (linhas ~487-525)
   - Copiar **literalmente** a lógica do try/except
   - Criar função com assinatura:

   ```python
   def compute_indecision_metrics_safe(
       analyzer: BERTAnalyzer,
       sbert_enabled: bool,
       sales_category: Optional[str],
       sales_category_confidence: Optional[float],
       sales_category_intensity: Optional[float],
       sales_category_ambiguity: Optional[float],
       conditional_keywords_detected: List[str],
       meeting_id: str = ""  # Para logs, se necessário
   ) -> Dict[str, Any]:
   ```

   **Regras de paridade (não negocie):**
   - [ ] Inicializar `indecision_metrics = {}` antes do try
   - [ ] Gating: `if sbert_enabled and sales_category is not None`
   - [ ] Chamar `analyzer.calculate_indecision_metrics(...)` com:
     - `sales_category_confidence or 0.0`
     - `sales_category_intensity or 0.0`
     - `sales_category_ambiguity or 0.0`
     - `conditional_keywords_detected`
   - [ ] Try/except que retorna `{}` em caso de erro (não propaga)
   - [ ] Retornar `{}` quando gating não passa

3. **Adicionar imports:**
   ```python
   from typing import Dict, Any, List, Optional
   from ..models.bert_analyzer import BERTAnalyzer
   import structlog
   ```

**IMPORTANTE:** Não mover `calculate_indecision_metrics` do `BERTAnalyzer` agora (fica onde está).

- Motivo: é mais invasivo e aumenta chance de regressão
- O ganho de separação já acontece ao tirar o bloco do `analysis_service.py`

---

### Passo 4: Integrar no Analysis Service

**Objetivo:** Substituir código inline por chamadas aos novos módulos, mantendo o mesmo fluxo.

**Arquivo:** `apps/text-analysis/src/services/analysis_service.py`

**Tarefas:**

#### 4.1 Adicionar Imports

No topo do arquivo, adicionar:

```python
from ..signals.reformulation import (
    detect_reformulation_markers,
    compute_reformulation_marker_score,
    apply_solution_reformulation_signal_flag
)
from ..signals.indecision import compute_indecision_metrics_safe
```

#### 4.2 Substituir Bloco de Reformulação

**Localização:** Dentro de `analyze()`, após classificação SBERT (linha ~475)

**Antes:**
```python
reformulation_markers_detected = self._detect_reformulation_markers(chunk.text)
reformulation_marker_score = min(1.0, len(reformulation_markers_detected) / 2.0)
if reformulation_marker_score > 0.0:
    sales_category_flags['solution_reformulation_signal'] = True
```

**Depois:**
```python
reformulation_markers_detected = detect_reformulation_markers(chunk.text)
reformulation_marker_score = compute_reformulation_marker_score(reformulation_markers_detected)
apply_solution_reformulation_signal_flag(sales_category_flags, reformulation_marker_score)
```

**Checklist de paridade:**
- [ ] `reformulation_markers_detected` continua sendo uma lista
- [ ] `reformulation_marker_score` continua sendo float entre 0..1
- [ ] `sales_category_flags` continua recebendo `solution_reformulation_signal=True` nos mesmos casos

#### 4.3 Substituir Bloco de Indecisão

**Localização:** Após bloco de Reformulação (linha ~487)

**Antes:**
```python
indecision_metrics: Dict[str, Any] = {}
try:
    if Config.SBERT_MODEL_NAME and sales_category is not None:
        indecision_metrics = analyzer.calculate_indecision_metrics(
            sales_category,
            sales_category_confidence or 0.0,
            sales_category_intensity or 0.0,
            sales_category_ambiguity or 0.0,
            conditional_keywords_detected
        )
        if indecision_metrics:
            logger.debug(...)
except Exception as e:
    logger.warn(...)
```

**Depois:**
```python
sbert_enabled = bool(Config.SBERT_MODEL_NAME)
indecision_metrics = compute_indecision_metrics_safe(
    analyzer=analyzer,
    sbert_enabled=sbert_enabled,
    sales_category=sales_category,
    sales_category_confidence=sales_category_confidence,
    sales_category_intensity=sales_category_intensity,
    sales_category_ambiguity=sales_category_ambiguity,
    conditional_keywords_detected=conditional_keywords_detected,
    meeting_id=chunk.meetingId
)
if indecision_metrics:
    logger.debug(
        "✅ [ANÁLISE] Métricas de indecisão calculadas",
        meeting_id=chunk.meetingId,
        indecision_score=round(indecision_metrics.get('indecision_score', 0.0), 4),
        postponement_likelihood=round(indecision_metrics.get('postponement_likelihood', 0.0), 4),
        conditional_language_score=round(indecision_metrics.get('conditional_language_score', 0.0), 4)
    )
```

**Checklist de paridade:**
- [ ] Quando não calcular, retorna `{}` (e o result continua gravando `None`)
- [ ] Quando calcular, retorna dict com chaves idênticas
- [ ] Em caso de exception, não quebra
- [ ] Logs de debug são mantidos (se existirem)

#### 4.4 Verificar Ordem de Execução

**Ordem correta (não alterar):**

1. Classificar categoria SBERT (gera `sales_category_flags` inicial)
2. Calcular reformulação + aplicar flag em `sales_category_flags`
3. Calcular indecisão
4. Contexto conversacional + agregações
5. `self.metrics.record_classification(... flags=sales_category_flags ...)`
6. Montar `result` dict

**CRÍTICO:** Flag de reformulação deve ser aplicada ANTES de `record_classification`.

- Se mover `apply_solution_reformulation_signal_flag(...)` para depois do `record_classification`, o contador de flags em `SemanticMetrics` muda

---

### Passo 5: Limpeza

**Objetivo:** Remover código duplicado ou manter compatibilidade.

**Decisão:** Remover ou manter método `_detect_reformulation_markers`

#### Opção A (Recomendada): Remover Método

**Prós:**
- Reduz duplicação e risco de divergência futura

**Contras:**
- Diff maior (mas seguro porque é método privado)

**Como fazer:**
1. Verificar que não há mais chamadas: `grep -r "_detect_reformulation_markers" apps/text-analysis/`
2. Se não houver outras referências além da definição, remover o método privado

#### Opção B (Conservadora): Manter Método Delegando

Se houver outras referências ou quiser manter compatibilidade:

Manter método, mas fazer dele delegar:
```python
def _detect_reformulation_markers(self, text: str) -> List[str]:
    from ..signals.reformulation import detect_reformulation_markers
    return detect_reformulation_markers(text)
```

**Prós:**
- Diff menor
- Mantém compatibilidade se alguém ainda chama o método por engano

**Contras:**
- Ainda existe um ponto "duplicado" (método + função)

---

### Passo 6: Testes

**Objetivo:** Provar que nada mudou após a refatoração.

#### 6.1 Testes Unitários para `reformulation.py`

**Arquivo:** `apps/text-analysis/tests/test_reformulation_signals.py`

**Casos de teste obrigatórios:**

- [ ] `test_detect_markers_encontra_marcadores`: input com marcadores → retorna lista com marcadores na ordem correta
- [ ] `test_detect_markers_nao_encontra`: input sem marcadores → retorna `[]`
- [ ] `test_detect_markers_mantem_ordem`: verificar que ordem do retorno segue ordem do array `markers`
- [ ] `test_detect_markers_case_insensitive`: verificar que `"DEIXA EU VER"` funciona
- [ ] `test_compute_score_vazio`: `[]` → `0.0`
- [ ] `test_compute_score_um_marcador`: `[marker1]` → `0.5`
- [ ] `test_compute_score_dois_marcadores`: `[marker1, marker2]` → `1.0`
- [ ] `test_compute_score_tres_mais`: `[marker1, marker2, marker3]` → `1.0` (cap)
- [ ] `test_apply_flag_score_zero`: score `0.0` → dict não é modificado
- [ ] `test_apply_flag_score_positivo`: score `> 0.0` → `flags['solution_reformulation_signal'] = True`

#### 6.2 Testes Unitários para `indecision.py`

**Arquivo:** `apps/text-analysis/tests/test_indecision_signals.py`

**Casos de teste obrigatórios:**

- [ ] `test_gating_sbert_disabled`: `sbert_enabled=False` → retorna `{}` e não chama analyzer
- [ ] `test_gating_category_none`: `sales_category=None` → retorna `{}` e não chama analyzer
- [ ] `test_gating_passa_chama_analyzer`: ambos habilitados → chama `analyzer.calculate_indecision_metrics()`
- [ ] `test_defaults_none_vira_zero`: `confidence=None` → passa `0.0` para analyzer
- [ ] `test_exception_retorna_vazio`: analyzer lança exception → retorna `{}` e não propaga
- [ ] `test_retorna_dict_correto`: analyzer retorna dict → função retorna mesmo dict

**Usar `Mock()` para `analyzer.calculate_indecision_metrics`.**

#### 6.3 Teste de Integração (Contrato do `analyze()`)

**Arquivo:** Adicionar em arquivo de testes existente ou criar novo

**Objetivo:** Provar que `TextAnalysisService.analyze()` retorna mesmos campos e valores.

**Setup:**
- Mockar `BERTAnalyzer.classify_sales_category()` retornando flags base `{}`
- Mockar `BERTAnalyzer.detect_conditional_keywords()`
- Mockar `BERTAnalyzer.generate_semantic_embedding()`
- Mockar `BERTAnalyzer.calculate_indecision_metrics()` retornando dict de exemplo

**Validações:**
- [ ] `analysis['reformulation_markers_detected']` existe e tem tipo correto
- [ ] `analysis['reformulation_marker_score']` existe e está entre 0.0 e 1.0
- [ ] Quando há marcadores: `analysis['sales_category_flags']['solution_reformulation_signal'] == True`
- [ ] `analysis['indecision_metrics']` é `dict` ou `None` (conforme gating)
- [ ] Quando `sales_category` presente: `indecision_metrics` é dict (não None)

**Executar testes:**
```bash
cd apps/text-analysis
pytest -v
```

---

### Passo 7: Verificação Manual

**Objetivo:** Validar comportamento em ambiente real (opcional, mas recomendado).

**Tarefas:**

1. **Iniciar serviço Python localmente**

2. **Fazer POST em `/analyze` (endpoint REST de debug)**

   **Cenários para testar:**

   **Cenário 1: Texto com marcador de reformulação**
   - Input: `"Deixa eu ver se entendi, então vocês fazem X e Y?"`
   - Verificar: `reformulation_markers_detected` contém marcadores
   - Verificar: `reformulation_marker_score > 0.0`
   - Verificar: `sales_category_flags.solution_reformulation_signal == True`

   **Cenário 2: Texto sem marcador**
   - Input: `"Ok, entendi"`
   - Verificar: `reformulation_markers_detected == []`
   - Verificar: `reformulation_marker_score == 0.0`
   - Verificar: `solution_reformulation_signal` não existe (ou não está presente)

   **Cenário 3: Texto com sales_category**
   - Input: texto que resulte em `sales_category` não-None
   - Verificar: `indecision_metrics` é dict (não None)

   **Cenário 4: Texto sem sales_category**
   - Input: texto que resulte em `sales_category` None
   - Verificar: `indecision_metrics` é None ou ausente

---

## Checklist Final

### Definition of Done (DoD)

- [ ] `src/signals/reformulation.py` existe com todas as funções
- [ ] `src/signals/indecision.py` existe com `compute_indecision_metrics_safe`
- [ ] `analysis_service.py` importa e usa os módulos
- [ ] Ordem de execução preservada (flag antes de `record_classification`)
- [ ] Payload final mantém todas as chaves:
  - [ ] `analysis.indecision_metrics`
  - [ ] `analysis.reformulation_markers_detected`
  - [ ] `analysis.reformulation_marker_score`
  - [ ] `analysis.sales_category_flags.solution_reformulation_signal`
- [ ] Todos os testes passam (existentes + novos)
- [ ] Método `_detect_reformulation_markers` removido ou delegando (sem duplicação divergente)
- [ ] Não há outras referências ao método privado (se removido)
- [ ] Código segue padrões do projeto (imports, estilo, etc)

---

## Observações Importantes

### ⚠️ Armadilhas Comuns (NÃO FAÇA)

- ❌ **NÃO transforme substring em regex**: `if m in t` deve continuar substring
- ❌ **NÃO normalize acentos**: `"só pra confirmar"` deve continuar com acento
- ❌ **NÃO altere tipo de retorno**: `indecision_metrics` continua `None` no payload quando dict vazio
- ❌ **NÃO altere condição da flag**: usar `marker_score > 0.0`, não `len(markers) > 0`
- ❌ **NÃO mude a ordem de efeitos**: flag deve ser aplicada antes de `record_classification`
- ❌ **NÃO mova `calculate_indecision_metrics`** do `BERTAnalyzer` neste passo (deixa para depois, se necessário)

---

## 📝 Notas Adicionais

### Por que separar?

- **Manutenibilidade**: Cada módulo tem responsabilidade única
- **Testabilidade**: Fácil testar cada análise isoladamente
- **Reutilização**: Funções podem ser reutilizadas em outros contextos
- **Code review**: Mudanças ficam mais focadas e fáceis de revisar

### Próximos Passos (Após Esta Refatoração)

- Considerar mover `calculate_indecision_metrics` para `signals/indecision.py` (separação mais completa)
- Adicionar mais análises em `signals/` (ex.: detecção de urgência, sentimento avançado)
- Criar interfaces/contratos mais explícitos (TypedDict, Pydantic models)

