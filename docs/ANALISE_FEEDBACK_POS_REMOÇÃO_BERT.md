# Análise: motivo do feedback não ser retornado após remoção do BERT

## Objetivo

Identificar, por inspeção do código (não por suposição), se as alterações feitas na remoção do BERT podem ser a **causa** do feedback não ser retornado.

---

## 1. Alterações feitas na remoção do BERT

| Arquivo | Alteração |
|---------|-----------|
| `src/models/bert_analyzer.py` | Removidos imports (AutoTokenizer, AutoModelForSequenceClassification, pipeline). `__init__` só recebe `sbert_model_name`. Removido `_load_model()`. `analyze_sentiment()` retorna fixo `{positive:0, negative:0, neutral:1}`. |
| `src/services/service_text_analysis/analysis/analysis_service.py` | `_get_analyzer()` instancia `BERTAnalyzer(sbert_model_name=...)` apenas. Removido bloco de carregamento do BERT em `_ensure_models_loaded()`. Removido `_load_bert_model_sync()`. Chamada a `_ensure_models_loaded(require_sbert=True)` no início de `analyze()`. |
| `src/services/service_text_analysis/analysis/semantic_pipeline.py` | Comentário em `analyze_sentiment`; lógica inalterada (continua chamando `analyzer.analyze_sentiment()`, que agora retorna neutro fixo). |
| `src/config.py` | Removidos `MODEL_NAME`, `MODEL_CACHE_DIR`, `MODEL_DEVICE`, `ANALYSIS_MAX_LENGTH` de `_Settings` e de `Config`. Removidos asserts em `validate()`. |
| `src/main.py` | Log de inicialização sem `model`/`device`; descrição do FastAPI. |
| `src/socketio_server.py` | Apenas comentários (BERT → SBERT). |
| `scripts/validate_sales_category.py` | Uso de `BERTAnalyzer(sbert_model_name=...)` apenas. |
| `tests/test_sales_category_classification.py` | Fixtures com nova assinatura de `BERTAnalyzer`. |

Nenhuma alteração foi feita em: `TranscriptionService`, `transcribe_audio()`, buffer de áudio, ou na condição que decide se há texto para analisar.

---

## 2. Fluxo de dados até o feedback

```
Áudio agregado → on_buffer_ready()
  → transcription_service.transcribe_audio(wav_data, ...)
  → transcription_result = { 'text': str, 'confidence': float, ... }
  → text = transcription_result.get('text', '').strip()
  → [SE text vazio] logger.warn("Nenhum texto transcrito..."); return   ← FIM (não há análise nem emit)
  → [SE text não vazio] chunk = TranscriptionChunk(...)
  → analysis_result = await analysis_service.analyze(chunk)
  → result_dict = TextAnalysisResult(...).model_dump()
  → dedupe/rate_limit (podem fazer return sem emitir)
  → emit (Redis ou Socket.IO) text_analysis_result
```

O backend NestJS recebe `text_analysis_result` e aí roda os detectores (ex.: `detect-client-indecision`) e emite o feedback. Portanto: **se o serviço Python não enviar `text_analysis_result`, o backend não pode devolver feedback.**

---

## 3. Onde cada alteração atua (ou não)

### 3.1 Transcrição → texto

- **Código:** `transcription_result = await transcription_service.transcribe_audio(...)` e `text = transcription_result.get('text', '').strip()`.
- **Alterações na remoção do BERT:** nenhuma nesse trecho nem em `TranscriptionService`.
- **Conclusão:** a decisão “texto vazio ou não” **não foi alterada** pela remoção do BERT.

### 3.2 Ramo “sem texto”

- **Código:** `if not text: logger.warn(...); return`.
- **Alterações:** nenhuma nessa condição.
- **Conclusão:** quando o log "Nenhum texto transcrito do áudio agrupado" aparece, o fluxo **sai aqui** e não chama `analyze()` nem emite. Isso já era assim antes da remoção do BERT.

### 3.3 Análise (quando há texto)

- **Código:** `analysis_result = await analysis_service.analyze(chunk)`.
- **Dentro de `analyze()`:**  
  `_ensure_models_loaded(require_sbert=True)` → `_get_analyzer()` → `run_semantic_pipeline(chunk, analyzer, self)` → `registry.run_all(...)` → montagem do `result` e `return result`.
- **Referências a config/atributos removidos:**
  - Em `analysis_service.py`: só `Config.SBERT_MODEL_NAME` (mantido). Não há uso de `MODEL_NAME`, `MODEL_DEVICE`, `MODEL_CACHE_DIR` ou `ANALYSIS_MAX_LENGTH`.
  - Em `semantic_pipeline.run()`: só `Config.SBERT_MODEL_NAME` e `analyzer.analyze_sentiment()` (agora retorno fixo, sem acessar modelo BERT).
  - Em `bert_analyzer`: não há mais uso de `self._loaded` (BERT); só `_sbert_loaded` para SBERT.
- **Conclusão:** quando há texto, o caminho que leva ao `return result` **não depende** de nenhum config ou atributo que tenhamos removido. Se `analyze()` falhar por causa das nossas alterações, a exceção seria capturada em `on_buffer_ready` e apareceria o log "❌ [BUFFER] Erro ao processar buffer pronto" (não apenas "Nenhum texto transcrito").

### 3.4 Formato do resultado e emit

- **Código:** `TextAnalysisResult(..., analysis=analysis_result)` e depois `result.model_dump()` / Redis ou Socket.IO.
- **Alterações:** o dicionário retornado por `analyze()` mantém as mesmas chaves; só a origem de `sentiment`/`sentiment_score`/`urgency` passou a ser o retorno fixo de `analyze_sentiment()`. O backend usa, para o feedback de indecisão, campos como `indecision_metrics`, `sales_category`, `sales_category_intensity`, `textHistory` — todos continuam produzidos pelo SBERT e pelos signals, sem BERT.
- **Conclusão:** o payload de `text_analysis_result` **não foi quebrado** pela remoção do BERT; não há motivo, por causa das nossas mudanças, para o backend deixar de emitir feedback quando recebe o evento.

---

## 4. Conclusão

- **Nenhuma alteração da remoção do BERT** modifica:
  - o fato de a transcrição poder devolver texto vazio;
  - a condição `if not text: return`;
  - o caminho que, na presença de texto, chama `analyze()` e emite `text_analysis_result`.
- O log que você viu — **"Nenhum texto transcrito do áudio agrupado"** — é impresso **nesse ramo** (`if not text`). Ou seja, nessa execução **não houve texto**, portanto **não houve chamada a `analyze()` e não houve envio de `text_analysis_result`**, e o backend não tem evento para gerar feedback.
- **Causa identificada (pelo código):** o feedback não é retornado porque, nos casos em que esse log aparece, **a transcrição está devolvendo texto vazio**. A remoção do BERT não altera esse comportamento; a causa não é as alterações do BERT e sim o fato de, nesse fluxo, não existir texto transcrito para analisar e enviar.

---

## 5. Próximos passos sugeridos

1. **Confirmar o diagnóstico:** em ambiente onde o problema ocorre, verificar se há **algum** log de "📤 [EMIT] text_analysis_result enviado ao backend". Se nunca aparecer, o Python realmente não está emitindo (por não ter texto ou por dedupe/rate limit). Se aparecer e ainda assim não houver feedback, o problema está no backend ou na regra do detector.
2. **Entender por que o texto está vazio:** usar os campos já (ou a serem) logados quando `text` vem vazio — por exemplo `has_speech`, `rejection_reason`, `segments_count` no resultado da transcrição — para ver se o áudio está sendo rejeitado no pré-processamento (ex.: RMS/“sem fala”) ou se o Whisper está retornando segmentos vazios.
3. **Testar com texto garantido:** enviar um `transcription_chunk` (caminho egress) com texto preenchido e conferir se o backend emite feedback; isso isola “transcrição vazia” de “análise/emit/backend”.
