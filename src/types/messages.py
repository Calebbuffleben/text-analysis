from typing import Optional, Dict, Any
from pydantic import BaseModel

class TranscriptionChunk(BaseModel):
    """Mensagem recebida do backend (texto já transcrito)"""
    meetingId: str
    participantId: str
    text: str
    timestamp: int
    language: Optional[str] = None
    confidence: Optional[float] = None

class AudioChunk(BaseModel):
    """Mensagem recebida do backend (áudio WAV para transcrição)"""
    meetingId: str
    participantId: str
    track: str
    audioData: bytes  # Dados WAV em base64 ou bytes
    sampleRate: int
    channels: int
    timestamp: int
    language: Optional[str] = None

class TextAnalysisResult(BaseModel):
    """Mensagem enviada para o backend"""
    meetingId: str
    participantId: str
    text: str
    analysis: Dict[str, Any]
    timestamp: int
    confidence: float

