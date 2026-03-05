"""
Pipeline semântico: sentiment, keywords, sales_category, contexto, agregação.
Extraído do orquestrador para manter analysis_service.py ≤150 linhas (spec §7).
"""

import re
from typing import Any, Dict, List, Tuple

import structlog

from ....config import Config
from ....models.conversation_context import ConversationContext
from ....types.messages import TranscriptionChunk

logger = structlog.get_logger()


def _detect_intent(text: str, has_question: bool) -> Tuple[str, float]:
    text_lower = text.lower()
    intent_patterns = {
        "ask_price": ["quanto", "custa", "valor", "preço", "price"],
        "ask_info": ["o que", "como", "quando", "onde", "quem"],
        "request_action": ["pode", "poderia", "favor", "por favor", "faça"],
        "express_opinion": ["acho", "penso", "acredito", "opinião"],
        "express_agreement": ["concordo", "sim", "exato", "certo"],
        "express_disagreement": ["discordo", "não", "errado", "incorreto"],
    }
    for intent, patterns in intent_patterns.items():
        if any(p in text_lower for p in patterns):
            matches = sum(1 for p in patterns if p in text_lower)
            return (intent, min(0.9, 0.5 + matches * 0.1))
    return ("ask_question", 0.6) if has_question else ("statement", 0.5)


def _split_for_sales_category(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[.!?;:\n]+", text) if p.strip()]
    segments = [p for p in parts if len(p.split()) >= 3 and len(p) >= 12]
    return segments[-6:] if len(segments) > 6 else segments


def _detect_topic(text: str, keywords: List[str]) -> Tuple[str, float]:
    text_lower = text.lower()
    topic_patterns = {
        "pricing": ["preço", "valor", "custo", "price", "quanto"],
        "product": ["produto", "serviço", "solução", "oferta"],
        "support": ["suporte", "ajuda", "problema", "erro", "bug"],
        "schedule": ["agendar", "horário", "data", "reunião", "meeting"],
        "technical": ["técnico", "implementação", "código", "tecnologia"],
    }
    for topic, patterns in topic_patterns.items():
        if any(p in text_lower for p in patterns) or any(kw in patterns for kw in keywords):
            matches = sum(1 for p in patterns if p in text_lower or p in keywords)
            return (topic, min(0.95, 0.6 + matches * 0.1))
    return ("general", 0.5)


def _detect_speech_act(text: str, has_question: bool, has_exclamation: bool) -> Tuple[str, float]:
    text_lower = text.lower()
    if has_question:
        return ("question", 0.9)
    if has_exclamation:
        return ("exclamation", 0.85)
    if any(p in text_lower for p in ["favor", "por favor", "pode", "poderia", "faça", "execute"]):
        return ("request", 0.8)
    if any(w in text_lower for w in ["sim", "certo", "ok", "entendi", "concordo"]):
        return ("agreement", 0.75)
    if any(w in text_lower for w in ["não", "discordo", "errado", "incorreto"]):
        return ("disagreement", 0.75)
    return ("statement", 0.7)


def _extract_entities(text: str, keywords: List[str]) -> List[str]:
    text_lower = text.lower()
    entity_patterns = {
        "preço": ["preço", "valor", "custo", "price"],
        "produto": ["produto", "serviço", "solução"],
        "data": ["hoje", "amanhã", "semana", "mês", "ano"],
        "pessoa": ["você", "eu", "nós", "eles"],
    }
    entities = []
    for _, patterns in entity_patterns.items():
        if any(p in text_lower for p in patterns):
            entities.extend(p for p in patterns if p in text_lower)
    for kw in keywords[:3]:
        if kw not in entities and len(kw) > 3:
            entities.append(kw)
    return list(dict.fromkeys(entities))[:5]


def _calculate_urgency(
    sentiment_score: float, has_question: bool, has_exclamation: bool, emotions: Dict[str, float]
) -> float:
    u = 0.5
    if has_question:
        u += 0.15
    if has_exclamation:
        u += 0.1
    u += (emotions.get("anger", 0.0) + emotions.get("fear", 0.0)) * 0.2
    if sentiment_score < 0.4:
        u += 0.1
    return min(1.0, max(0.0, u))


async def run(
    chunk: TranscriptionChunk, analyzer: Any, svc: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executa pipeline semântico (sentiment → sales_category → contexto → agregação).
    Retorna (base_dict, semantic_result) para o orquestrador fazer dispatch e merge.
    Sentimento: neutro fixo (analyze_sentiment retorna sem modelo de sentimento).
    """
    sentiment = analyzer.analyze_sentiment(chunk.text)  # neutro fixo: {positive:0, negative:0, neutral:1}
    keywords = analyzer.extract_keywords(chunk.text, top_n=10)
    emotions = analyzer.detect_emotions(chunk.text)
    if Config.SBERT_MODEL_NAME:
        await svc._ensure_models_loaded(require_sbert=True)
        try:
            analyzer.analyze_semantics(chunk.text)
        except Exception as e:
            logger.warn("semantic_analysis_failed", error=str(e), meeting_id=chunk.meetingId)
    word_count = len(chunk.text.split())
    char_count = len(chunk.text)
    has_question = "?" in chunk.text
    has_exclamation = "!" in chunk.text
    sentiment_label = "neutral"
    sentiment_single_score = sentiment.get("neutral", 0.0)
    if sentiment.get("positive", 0.0) > sentiment.get("negative", 0.0) and sentiment.get("positive", 0.0) > sentiment.get("neutral", 0.0):
        sentiment_label, sentiment_single_score = "positive", sentiment.get("positive", 0.0)
    elif sentiment.get("negative", 0.0) > sentiment.get("neutral", 0.0):
        sentiment_label, sentiment_single_score = "negative", sentiment.get("negative", 0.0)
    intent, intent_confidence = _detect_intent(chunk.text, has_question)
    topic, topic_confidence = _detect_topic(chunk.text, keywords)
    speech_act, speech_act_confidence = _detect_speech_act(chunk.text, has_question, has_exclamation)
    entities = _extract_entities(chunk.text, keywords)
    urgency = _calculate_urgency(sentiment_single_score, has_question, has_exclamation, emotions)
    conditional_keywords_detected: List[str] = []
    if Config.SBERT_MODEL_NAME:
        try:
            conditional_keywords_detected = analyzer.detect_conditional_keywords(chunk.text, keywords)
        except Exception as e:
            logger.warn("conditional_keywords_failed", error=str(e), meeting_id=chunk.meetingId)
    embedding: List[float] = []
    text_for_embedding = (chunk.text or "").strip()
    if Config.SBERT_MODEL_NAME and text_for_embedding:
        def _do_embedding() -> List[float]:
            arr = analyzer.generate_semantic_embedding(chunk.text)
            if arr is None or len(arr) == 0:
                return []
            flat = arr.ravel() if hasattr(arr, "ravel") else arr
            return [float(x) for x in flat]

        for attempt in range(2):
            try:
                await svc._ensure_models_loaded(require_sbert=True)
                embedding = _do_embedding()
                if embedding:
                    break
                logger.warn(
                    "embedding_empty",
                    message="generate_semantic_embedding returned empty or None",
                    meeting_id=chunk.meetingId,
                    text_len=len(chunk.text),
                    attempt=attempt + 1,
                )
            except Exception as e:
                logger.error(
                    "embedding_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    meeting_id=chunk.meetingId,
                    text_len=len(chunk.text),
                    attempt=attempt + 1,
                )

        if not embedding and Config.SBERT_MODEL_NAME and text_for_embedding:
            logger.error(
                "embedding_missing",
                message="Embedding empty after retries; solution_understood feedback will not fire for this segment",
                meeting_id=chunk.meetingId,
                participant_id=chunk.participantId,
                text_len=len(chunk.text),
            )
    sales_category = None
    sales_category_confidence = None
    sales_category_ambiguity = None
    sales_category_intensity = None
    sales_category_flags: Dict[str, bool] = {}
    sales_category_best_score = 0.0
    sales_category_scores: Dict[str, float] = {}
    sales_category_top_3: List[Dict[str, float]] = []
    if Config.SBERT_MODEL_NAME:
        await svc._ensure_models_loaded(require_sbert=True)
        try:
            cat, conf, scores, amb, intensity, flags = analyzer.classify_sales_category(
                chunk.text, min_confidence=0.15
            )
            sales_category, sales_category_confidence = cat, conf
            sales_category_ambiguity, sales_category_intensity = amb, intensity
            sales_category_flags = flags
            sales_category_best_score = max(scores.values()) if scores else 0.0
            sales_category_scores = scores
            sales_category_top_3 = [
                {"category": c, "score": round(s, 4)}
                for c, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            ]
            if sales_category is None:
                segments = _split_for_sales_category(chunk.text)
                if len(segments) > 1:
                    best_score, best_res = 0.0, None
                    for seg in segments:
                        seg_cat, seg_conf, seg_scores, seg_amb, seg_int, seg_flags = analyzer.classify_sales_category(
                            seg, min_confidence=0.0
                        )
                        if not (seg_scores and len(seg_scores)):
                            continue
                        sb = max(seg_scores.values())
                        if sb > best_score:
                            best_score, best_res = sb, (
                                seg_cat, seg_conf, seg_scores, seg_amb, seg_int, seg_flags,
                            )
                    if best_res:
                        sc, _, seg_scores, samb, sint, sfl = best_res
                        if sc in ("stalling", "objection_soft") and best_score >= 0.15 and (sint or 0.0) >= 0.25:
                            sales_category, sales_category_ambiguity = sc, samb
                            sales_category_intensity, sales_category_flags = sint, sfl
                            sales_category_confidence = max(seg_scores.values()) if seg_scores else best_score
                            sales_category_best_score = best_score
                            sales_category_scores = seg_scores
                            sales_category_top_3 = [
                                {"category": c, "score": round(s, 4)}
                                for c, s in sorted(seg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                            ]
        except Exception as e:
            logger.warn("sales_category_failed", error=str(e), meeting_id=chunk.meetingId)
    semantic_result = {
        "sales_category": sales_category,
        "sales_category_confidence": sales_category_confidence,
        "sales_category_intensity": sales_category_intensity,
        "sales_category_ambiguity": sales_category_ambiguity,
        "sales_category_flags": sales_category_flags,
        "conditional_keywords_detected": conditional_keywords_detected,
    }
    context_key = svc._get_context_key(chunk)
    if context_key not in svc.conversation_contexts:
        svc.conversation_contexts[context_key] = ConversationContext(
            window_size=svc.context_window_size,
            window_duration_ms=svc.context_window_duration_ms,
        )
    ctx = svc.conversation_contexts[context_key]
    ctx.add_chunk({
        "text": chunk.text,
        "sales_category": sales_category,
        "sales_category_confidence": sales_category_confidence,
        "sales_category_intensity": sales_category_intensity,
        "sales_category_ambiguity": sales_category_ambiguity,
        "timestamp": chunk.timestamp,
        "embedding": embedding,
    })
    window = ctx.get_window(chunk.timestamp)
    sales_category_aggregated = None
    sales_category_transition = None
    sales_category_trend = None
    if window and Config.SBERT_MODEL_NAME:
        try:
            sales_category_aggregated = analyzer.aggregate_categories_temporal(window)
            if sales_category and sales_category_confidence is not None:
                sales_category_transition = analyzer.detect_category_transition(
                    sales_category, sales_category_confidence, window
                )
            sales_category_trend = analyzer.calculate_semantic_trend(window)
        except Exception as e:
            logger.warn("contextual_analysis_failed", error=str(e), meeting_id=chunk.meetingId)
    base = {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "topic": topic,
        "topic_confidence": topic_confidence,
        "speech_act": speech_act,
        "speech_act_confidence": speech_act_confidence,
        "keywords": keywords,
        "entities": entities,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_single_score,
        "urgency": urgency,
        "embedding": embedding,
        "sales_category": sales_category,
        "sales_category_confidence": sales_category_confidence,
        "sales_category_intensity": sales_category_intensity,
        "sales_category_ambiguity": sales_category_ambiguity,
        "sales_category_flags": sales_category_flags,
        "sales_category_best_score": sales_category_best_score,
        "sales_category_scores": sales_category_scores,
        "sales_category_top_3": sales_category_top_3,
        "sales_category_aggregated": sales_category_aggregated,
        "sales_category_transition": sales_category_transition,
        "sales_category_trend": sales_category_trend,
        "conditional_keywords_detected": conditional_keywords_detected,
        "_word_count": word_count,
        "_char_count": char_count,
    }
    return (base, semantic_result)
