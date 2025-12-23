## Planejamento Arquitetural: Detecção “Solução foi compreendida” (Reformulação do Cliente)

**Objetivo**: gerar um feedback em tempo real para o vendedor quando o **cliente reformula a solução com as próprias palavras** (sinal forte de compreensão).  
**Base semântica**: “frases de reformulação pelo cliente” (teach‑back / paraphrase).  
**Fluxo**: igual ao de indecisão (áudio → Python `text_analysis_result` → `FeedbackAggregatorService` → `FeedbackDeliveryService` → overlay na extensão).

---

## Checklist (passo a passo)

> Use esta lista como roteiro de implementação. As seções abaixo já estão organizadas nos mesmos passos, com detalhes.

- [ ] **Passo 0 — Confirmar o pipeline e dados disponíveis** (payload já tem `analysis.embedding`, keywords, speech_act).
- [ ] **Passo 1 — Definir o sinal e falsos positivos** (o que é “reformulação” vs “ok entendi”).
- [ ] **Passo 2 — Escolher base semântica e thresholds iniciais** (cosine similarity + ranges).
- [ ] **Passo 3 — Criar estado/memória curta no backend** (contexto de solução do host).
- [ ] **Passo 4 — Detectar “explicação de solução” do host** (context builder + strength).
- [ ] **Passo 5 — Detectar marcadores de reformulação do cliente** (regex/substring + “conteúdo suficiente”).
- [ ] **Passo 6 — Calcular similaridade e confidence combinado** (pesos + threshold).
- [ ] **Passo 7 — Gating + cooldown + antispam** (mesmo padrão do indecision).
- [ ] **Passo 8 — Definir payload/UX do feedback** (`severity`, message, tips, metadata).
- [ ] **Passo 9 — Implementar no backend (arquivos e integração)** (`handleTextAnalysis` + delivery).
- [ ] **Passo 10 (Opcional) — Melhorias no Python** (flags/métricas, se quiser).
- [ ] **Passo 11 — Testes e validação** (unit + golden dataset).
- [ ] **Passo 12 — Rollout e segurança** (feature flag, env vars, privacidade).
- [ ] **Passo 13 — Riscos e mitigação** (FPs, contexto errado, transcrição ruidosa).
- [ ] **Passo 14 — DoD** (critérios de pronto).

---

## Passo 0 — Confirmar o pipeline e dados disponíveis

- **Backend (Nest)** recebe eventos `text_analysis_result` via `TextAnalysisService` (`apps/backend/src/pipeline/text-analysis.service.ts`).
- O evento já contém:
  - `analysis.embedding: number[]` (SBERT, dimensão 384)
  - `analysis.keywords`, `analysis.speech_act`, `analysis.intent`, `analysis.sales_category_*` etc.
- O `FeedbackAggregatorService` (`apps/backend/src/feedback/feedback.aggregator.service.ts`) mantém estado por participante e dispara heurísticas (como `sales_client_indecision`).
- O `FeedbackDeliveryService` publica para os hosts via Socket.IO (room `feedback:${meetingId}`), exibido no overlay (`apps/chrome-extension/feedback-overlay.js`).

**Implicação importante**: dá para implementar “solução foi compreendida” **sem mudanças no serviço Python** no MVP, porque a semântica necessária (embedding + keywords + speech_act) já existe no payload.

---

## Passo 1 — Definição operacional do sinal (“cliente reformulou a solução”)

Chamaremos de **Reformulação do Cliente** um turno de fala do cliente que:

- **(A) É metacomunicativo de compreensão** (ex.: “entendi”, “se eu entendi bem”, “então é assim…”, “ou seja…”, “resumindo…”), e
- **(B) Reexpressa o mecanismo/benefício da solução** usando conteúdo (não apenas “ok entendi”), e
- **(C) Tem alta similaridade semântica** com a explicação de solução dita anteriormente pelo vendedor.

O feedback “solução foi compreendida” deve disparar quando a evidência \(A \land B \land C\) for forte o suficiente.

---

## Passo 2 — Base semântica (embeddings + cosine similarity)

“Reformulação” é, por natureza, **paráfrase**. Se o cliente compreende, ele tende a:

- manter o **mesmo significado**
- trocar **termos/sintaxe**

Embeddings semânticos capturam isso bem (ao contrário de matching literal por palavras).

Vamos usar **cosine similarity**:

\[
\text{cos\_sim}(u,v) = \frac{u\cdot v}{\|u\|\|v\|}
\]

Onde:
- \(u\) = embedding do turno do cliente (candidato)
- \(v\) = embedding de um **resumo do “contexto de solução”** recente do vendedor (centroide/mean pooling)

---

## Passo 3 — Estado (memória curta) necessária no backend

### 1) Identificar “quem é cliente” vs “host”

O feedback é para o vendedor/hosts, mas o sinal vem do cliente. Reusar a mesma regra já usada no fluxo de “sales_*”.

Recomendação:
- manter/usar método existente do `FeedbackAggregatorService`/`ParticipantIndexService` para determinar se um `participantId` é host.

### 2) Guardar contexto de “explicação de solução” do host

Crie uma estrutura de contexto por meeting (ou por host), com janela curta (ex.: últimos 90s):

- **`SolutionContextEntry`**:
  - `ts`
  - `text`
  - `embedding` (384 floats)
  - `keywords` (do Python)
  - `strength` (heurística: quão “explicação de solução” isso parece)

Política de retenção:
- manter últimos N (ex.: 12 entradas) e/ou últimos 90s.

### 3) Guardar candidatos do cliente

Não precisa persistir tudo; basta o turno atual (embedding + texto). O histórico só é útil para:
- consistência temporal
- evitar duplicação (cooldown)

---

## Passo 4 — Heurística: como detectar “explicação de solução” do host (contexto)

O maior risco arquitetural aqui é comparar reformulação do cliente com um “contexto errado”.

No MVP, usar heurística leve (explicável) para construir o contexto \(v\):

### Sinais para marcar um turno do host como “solution explanation”

- **Comprimento**: texto com pelo menos X caracteres (ex.: 80+) ou X tokens estimados.
- **Padrões linguísticos**:
  - “funciona assim…”
  - “na prática…”
  - “o fluxo é…”
  - “a gente faz…”
  - “você vai conseguir…”
  - “a solução resolve…”
- **Sales category compatível** (se disponível):
  - `value_exploration`, `information_gathering`, `price_interest` (depende do seu taxonomy atual)
  - evitar categorias de cliente (stalling/objection_soft) e ruído
- **Keywords de produto/ação**: interseção com um pequeno dicionário “solution-ish”

Defina um `strength` em [0,1] combinando esses sinais, e só inclua no contexto quando `strength >= 0.6`.

**Por que isso é suficiente no MVP**: o disparo final ainda exige alta similaridade semântica + marcadores explícitos de reformulação do cliente.

---

## Passo 5 — Heurística: como detectar “frases de reformulação” no turno do cliente

### 1) Marcadores de reformulação (lista inicial PT‑BR)

Crie um detector leve por regex/substring (sem NLP pesado):

- “se eu entendi”
- “entendi então”
- “entendi que”
- “então vocês”
- “ou seja”
- “resumindo”
- “na prática então”
- “quer dizer que”
- “basicamente”
- “só pra confirmar”
- “deixa eu ver se entendi”
- “então o que você está dizendo é”

Retornar:
- `markersDetected: string[]`
- `markerScore` (ex.: min(1, markersDetected.length / 2))

### 2) “Conteúdo suficiente” (evitar falso positivo de “ok entendi”)

Regras simples:
- texto do cliente precisa ter pelo menos `MIN_REFORMULATION_CHARS` (ex.: 30–50)
- e/ou pelo menos 1–2 keywords relevantes (interseção com keywords do contexto do host)

---

## Passo 6 — Cálculo de similaridade e confidence (MVP)

### 1) Construção do vetor de contexto do host

Selecione entradas no intervalo [now - 90s, now] com `strength >= 0.6`.  
Compute o centroide:

\[
v = \text{mean}(e_1, e_2, ..., e_n)
\]

Se \(n=0\), não há contexto → não disparar.

### 2) Similaridade

\[
s = \text{cos\_sim}(u, v)
\]

Valores típicos com SBERT:
- < 0.55: não relacionado
- 0.60–0.70: possivelmente relacionado (depende de ruído)
- > 0.72: fortemente relacionado (bom threshold inicial)

### 3) Confidence combinado

Proposta (explicável e ajustável):

- `similarityScore` = clamp((s - 0.55) / 0.25, 0, 1)
- `markerScore` = conforme marcadores
- `contextStrength` = média dos `strength` do contexto usado
- `keywordOverlapScore` = min(1, overlap / 3) (overlap = |KW_client ∩ KW_context|)
- `speechActScore`:
  - 1.0 se `speech_act` ∈ {`agreement`, `confirmation`}
  - 0.5 se `ask_info` (pode ser confirmação)
  - 0.0 caso contrário

Confidence final:
- 45% similarity
- 20% marker
- 15% keyword overlap
- 15% context strength
- 5% speech act

Threshold inicial recomendado:
- `SALES_SOLUTION_UNDERSTOOD_THRESHOLD=0.70`

---

## Passo 7 — Regras de disparo (gating) e antispam

### Gating mínimo (antes de calcular/confiança)

Só avalia o detector se:
- o falante é **cliente** (não host)
- `markersDetected.length > 0`
- texto do cliente >= `MIN_REFORMULATION_CHARS`
- existe contexto de solução do host na janela (>=1 entry)

### Cooldown

Evitar spam:
- `SALES_SOLUTION_UNDERSTOOD_COOLDOWN_MS` default 120000 (2 min)
- mesma regra de “cooldown desabilitado com 0” usada em indecisão pode ser reaproveitada.

### Dedupe semântico (opcional)

Mesmo dentro do cooldown, pode haver casos onde vale notificar “de novo” (ex.: reformulações de pontos diferentes). Para o MVP, **não** faça isso; mantenha simples.

---

## Passo 8 — Feedback gerado (payload e UX)

### Novo tipo

Adicionar `sales_solution_understood` (ou nome equivalente) como `FeedbackType`.

Recomendação:
- `severity: 'info'` (é um “bom sinal”, não alerta)

### Mensagem e dicas (curtas e acionáveis)

- **message (exemplos)**:
  - “Cliente reformulou sua solução — parece que entendeu.”
  - “Bom sinal: o cliente está ‘devolvendo’ o entendimento da solução.”

- **tips (exemplos)**:
  - “Confirme: ‘Perfeito — é isso mesmo.’”
  - “Valide o próximo passo: ‘Faz sentido avançarmos para X?’”
  - “Pergunte o critério: ‘O que falta para decidir?’”

### Metadata (para debug e evolução)

Salvar no `metadata`:
- `similarity_raw` (s)
- `confidence`
- `markers_detected`
- `keyword_overlap`
- `solution_context_excerpt` (1–2 frases do host)
- `client_reformulation_excerpt` (trecho do cliente)

---

## Passo 9 — Mudanças no backend (arquivos e pontos de integração)

### 1) `apps/backend/src/feedback/feedback.aggregator.service.ts`

- **Novo detector**:
  - `detectClientSolutionUnderstood(state, evt, now): FeedbackEventPayload | null`
- **Novo estado em memória**:
  - `state.textAnalysis.solutionContext` (ou estrutura por meeting) para guardar `SolutionContextEntry[]`
- **Funções auxiliares**:
  - `isHost(participantId)` (reuso)
  - `isSolutionExplanationHostTurn(evt)` → strength
  - `detectReformulationMarkers(text)` → markers + score
  - `cosineSimilarity(a,b)` e `meanEmbedding(list)`
  - `keywordOverlap(clientKeywords, contextKeywords)`

Integração:
- em `handleTextAnalysis(evt)` depois de atualizar o estado:
  - atualizar contexto se for host + explanation strength
  - rodar detector se for cliente
  - se `feedback != null` → `this.delivery.publishToHosts(...)`

### 2) `apps/backend/src/feedback/feedback.types.ts`

- adicionar o novo `type` no union/enum do payload.

### 3) Prisma / persistência (se salvar no DB)

- `apps/backend/prisma/schema.prisma`: adicionar valor no enum `FeedbackType`
- criar migração com `ALTER TYPE "FeedbackType" ADD VALUE ...`

---

## Passo 10 (Opcional) — Mudanças no serviço Python (“fase 2”)

O MVP não precisa, mas para melhorar precisão e reduzir heurísticas no backend:

- adicionar flag em `sales_category_flags`:
  - `solution_reformulation_signal?: boolean`
- adicionar métricas:
  - `reformulation_marker_score`
  - `reformulation_similarity_hint` (se o Python mantiver contexto — geralmente não vale a pena; melhor manter no backend)

**Recomendação**: manter contexto no backend é melhor porque:
- o backend já gerencia “meeting state” e cooldowns
- evita acoplar estado no serviço Python (escalabilidade e reinícios)

---

## Passo 11 — Testes e validação (nível “irretocável”)

### 1) Unit tests (backend)

Cobrir:
- `detectReformulationMarkers()` (positivos/negativos)
- `cosineSimilarity()` (vetores conhecidos)
- `meanEmbedding()` (edge cases)
- `keywordOverlap()`
- `detectClientSolutionUnderstood()` com cenários sintéticos:
  - sem contexto → null
  - contexto fraco → null
  - cliente “ok entendi” → null
  - cliente reformula com similaridade alta → feedback
  - cooldown respeitado / cooldown 0 desabilita

### 2) Golden dataset (offline)

Monte um conjunto rotulado com 3 classes:
- `reformulation_true` (cliente reexpressa solução corretamente)
- `reformulation_false` (marcadores sem conteúdo: “entendi”, “ok”)
- `other` (qualquer coisa)

Métricas:
- **precision** alta é prioridade (evitar spam)
- recall é secundário no MVP (pode melhorar depois)

### 3) Observabilidade em produção

Adicionar logs debug semelhantes ao indecision:
- gating reasons
- `similarity_raw`, `confidence`, `markers`, overlap

Adicionar contadores em `FeedbackDeliveryService.getMetrics(meetingId)`:
- `sales_solution_understood` count

---

## Passo 12 — Estratégia de rollout e segurança

- Feature flag:
  - `SALES_SOLUTION_UNDERSTOOD_ENABLED=true|false` (default: false em produção inicialmente)
- Threshold e cooldown configuráveis:
  - `SALES_SOLUTION_UNDERSTOOD_THRESHOLD` (default 0.70)
  - `SALES_SOLUTION_UNDERSTOOD_COOLDOWN_MS` (default 120000)
- Privacidade:
  - contexto/embeddings ficam apenas em memória (RAM), com janela curta
  - persistir no DB só o feedback, não o embedding bruto (evitar armazenar vetores)

---

## Passo 13 — Principais riscos e como mitigar

- **Falso positivo por “entendi”**:
  - mitigar com `MIN_REFORMULATION_CHARS` + overlap de keywords + threshold de similarity
- **Contexto errado (host falou outra coisa)**:
  - mitigar com `strength` e janela curta (90s) + exigir marcadores de reformulação
- **Transcrição ruidosa**:
  - exigir evidência redundante (markers + similarity + overlap)
  - reduzir sensibilidade em ambientes com baixa qualidade de áudio

---

## Passo 14 — Definição de pronto (DoD)

- detector gera `sales_solution_understood` em cenários realistas
- não gera em “ok entendi”/ruído
- cooldown evita spam
- métricas/logs permitem debugar rapidamente (thresholds, similarity)
- novo tipo compatível com DB (se persistência estiver ativa)


