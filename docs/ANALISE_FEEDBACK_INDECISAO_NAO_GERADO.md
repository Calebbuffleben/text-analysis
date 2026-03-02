# Análise: apenas o feedback de indecisão não é gerado

## Causas identificadas

### 1. Backend: filtro por origem do evento (env)

No agregador do backend (`feedback.aggregator.service.ts`), o feedback de indecisão pode ser **descartado** quando:

- `SALES_CLIENT_INDECISION_SOURCE_ONLY=buffer` está definido **e**
- o evento tem `source === 'egress'`.

Nesse caso o backend não publica o feedback e loga algo como:  
`🔇 [INDECISION_SOURCE] Dropping indecision feedback from egress (SALES_CLIENT_INDECISION_SOURCE_ONLY=buffer)`.

- **Buffer:** áudio agregado no Python → Whisper → análise → `result_dict['source'] = 'buffer'`.
- **Egress:** backend envia `transcription_chunk` para o Python → análise → `result_dict['source'] = 'egress'`.

**Como verificar:** Nos logs do backend, procure por `[INDECISION_SOURCE] Dropping indecision feedback from egress`. Se aparecer, o filtro está ativo e os eventos são egress.

**Ação:** Se os eventos de análise chegarem pelo fluxo egress, não defina `SALES_CLIENT_INDECISION_SOURCE_ONLY=buffer` (ou use outro valor). Caso contrário, o feedback de indecisão nunca será publicado para eventos egress.

---

### 2. Python: métricas de indecisão vazias quando não há `sales_category`

Em `src/signals/indecision.py`, `compute_indecision_metrics_safe` **só chama** `analyzer.calculate_indecision_metrics(...)` quando:

- `sbert_enabled` é verdadeiro **e**
- `sales_category is not None`.

Quando `sales_category` é `None` (SBERT não classificou ou confiança baixa), a função retorna `{}`. O backend recebe `indecision_metrics` vazio e:

- **Regra 1** (linguagem condicional): `(indecision_metrics?.conditional_language_score ?? 0) > 0.6` → sempre falso.
- **Regra 2** (postergar decisão): `(indecision_metrics?.postponement_likelihood ?? 0) > 0.6` → sempre falso.
- **Regra 3** (indecisão persistente): depende de `sales_category === 'stalling'` e `intensity > 0.5` no estado; se o chunk atual não for stalling, também não dispara.

Ou seja, quando não há categoria de vendas, as regras 1 e 2 nunca disparam e a 3 só dispara se o estado atual for stalling com intensidade alta.

**Correção aplicada:** Calcular e enviar ao menos `conditional_language_score` quando houver `conditional_keywords_detected`, mesmo com `sales_category` `None`, para que a regra 1 possa disparar a partir só de palavras condicionais.

---

### 3. Limiares das regras no backend

- **Regra 1:** `conditional_language_score > 0.6` → no Python hoje é `min(1.0, len(conditional_keywords)/5.0)`, então é preciso **pelo menos 4** keywords condicionais para passar (4/5 = 0.8).
- **Regra 2:** `postponement_likelihood > 0.6` → no Python só é preenchido para `category == 'stalling'`, com fórmula que tende a valores altos quando a intensidade é baixa; é um caso mais raro.
- **Regra 3:** `sales_category === 'stalling'` e `sales_category_intensity > 0.5` e (outro chunk com stalling nos últimos 20s ou janela vazia). Depende de o SBERT classificar como stalling com intensidade alta.

Se quiser que o feedback dispare com menos keywords condicionais, seria necessário ajustar o limiar no backend ou a fórmula no Python (por exemplo, normalizar por outro divisor).

---

## Resumo

| Causa | Onde | Ação |
|-------|------|------|
| Filtro por origem | Backend (env) | Não usar `SALES_CLIENT_INDECISION_SOURCE_ONLY=buffer` se os eventos forem egress. |
| Métricas vazias sem `sales_category` | Python `compute_indecision_metrics_safe` | Incluir `conditional_language_score` quando houver `conditional_keywords_detected` (correção aplicada). |
| Limiares altos | Backend + Python | Opcional: revisar limiares ou fórmulas se quiser mais disparos. |
