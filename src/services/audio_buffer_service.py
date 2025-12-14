"""
Serviço de buffer de áudio para agrupar chunks antes da transcrição.

Este serviço agrupa chunks de áudio pequenos em buffers maiores antes de enviar
para transcrição, melhorando:
- Performance: Whisper funciona melhor com áudio mais longo
- Qualidade: Transcrições mais precisas com contexto maior
- Eficiência: Menos chamadas ao Whisper
"""

import asyncio
import time
import structlog
from typing import Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
import wave
import io

logger = structlog.get_logger()


class AudioBuffer:
    """Buffer para armazenar chunks de áudio de um participante"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_chunks: list[bytes] = []
        self.first_timestamp: Optional[int] = None
        self.last_timestamp: Optional[int] = None
        self.total_samples = 0
        
    def add_chunk(self, wav_data: bytes, timestamp: int):
        """Adiciona um chunk WAV ao buffer"""
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        
        self.last_timestamp = timestamp
        self.audio_chunks.append(wav_data)
        
        # Calcular duração aproximada (WAV tem header de 44 bytes)
        if len(wav_data) > 44:
            # Tamanho dos dados de áudio (sem header)
            audio_data_size = len(wav_data) - 44
            # Cada sample é 2 bytes (16-bit PCM)
            samples = audio_data_size // 2
            self.total_samples += samples
    
    def get_duration_sec(self) -> float:
        """Retorna duração total do buffer em segundos"""
        if self.total_samples == 0:
            return 0.0
        return self.total_samples / self.sample_rate
    
    def get_combined_wav(self) -> bytes:
        """
        Combina todos os chunks WAV em um único buffer WAV.
        
        Estratégia:
        1. Decodifica cada chunk WAV para PCM
        2. Concatena os dados PCM
        3. Cria um novo WAV com os dados concatenados
        """
        if not self.audio_chunks:
            return b''
        
        # Decodificar todos os chunks para PCM
        all_pcm_data = []
        for wav_chunk in self.audio_chunks:
            try:
                wav_io = io.BytesIO(wav_chunk)
                with wave.open(wav_io, 'rb') as wav_file:
                    # Ler todos os frames
                    frames = wav_file.readframes(wav_file.getnframes())
                    all_pcm_data.append(frames)
            except Exception as e:
                logger.warn(
                    "⚠️ [BUFFER] Erro ao decodificar chunk WAV",
                    error=str(e),
                    chunk_size=len(wav_chunk)
                )
                continue
        
        if not all_pcm_data:
            return b''
        
        # Concatenar todos os dados PCM
        combined_pcm = b''.join(all_pcm_data)
        
        # Criar novo WAV com dados concatenados
        wav_output = io.BytesIO()
        with wave.open(wav_output, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(combined_pcm)
        
        return wav_output.getvalue()
    
    def clear(self):
        """Limpa o buffer"""
        self.audio_chunks.clear()
        self.first_timestamp = None
        self.last_timestamp = None
        self.total_samples = 0


class AudioBufferService:
    """
    Serviço para gerenciar buffers de áudio por participante.
    
    Agrupa chunks de áudio antes de enviar para transcrição, melhorando
    performance e qualidade.
    """
    
    def __init__(
        self,
        min_duration_sec: float = 3.0,  # Duração mínima antes de transcrever
        max_duration_sec: float = 10.0,  # Duração máxima (força transcrição)
        flush_interval_sec: float = 2.0,  # Intervalo máximo para flush
    ):
        """
        Inicializa serviço de buffer.
        
        Args:
            min_duration_sec: Duração mínima de áudio antes de transcrever (3s)
            max_duration_sec: Duração máxima antes de forçar transcrição (10s)
            flush_interval_sec: Tempo máximo entre chunks antes de flush (2s)
        """
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.flush_interval_sec = flush_interval_sec
        
        # Buffers por chave: (meeting_id, participant_id, track)
        self.buffers: Dict[Tuple[str, str, str], AudioBuffer] = {}
        
        # Timers para flush automático
        self.flush_timers: Dict[Tuple[str, str, str], asyncio.Task] = {}
        
        # Callback para quando buffer está pronto para transcrição
        self.on_buffer_ready = None
        
        logger.info(
            "✅ [BUFFER] AudioBufferService inicializado",
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            flush_interval_sec=flush_interval_sec
        )
    
    def set_callback(self, callback):
        """Define callback para quando buffer está pronto"""
        self.on_buffer_ready = callback
    
    async def add_chunk(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        wav_data: bytes,
        sample_rate: int,
        channels: int,
        timestamp: int,
    ) -> Optional[bytes]:
        """
        Adiciona chunk ao buffer e retorna áudio combinado se estiver pronto.
        
        Retorna:
            bytes: Áudio WAV combinado se buffer atingiu duração mínima, None caso contrário
        """
        key = (meeting_id, participant_id, track)
        
        # Criar buffer se não existir
        if key not in self.buffers:
            self.buffers[key] = AudioBuffer(sample_rate=sample_rate, channels=channels)
            logger.debug(
                "🆕 [BUFFER] Novo buffer criado",
                meeting_id=meeting_id,
                participant_id=participant_id,
                track=track
            )
        
        buffer = self.buffers[key]
        buffer.add_chunk(wav_data, timestamp)
        
        # Cancelar timer anterior se existir
        if key in self.flush_timers:
            self.flush_timers[key].cancel()
        
        duration = buffer.get_duration_sec()
        
        logger.debug(
            "📦 [BUFFER] Chunk adicionado ao buffer",
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            duration_sec=round(duration, 2),
            min_duration_sec=self.min_duration_sec,
            chunks_count=len(buffer.audio_chunks)
        )
        
        # Verificar se atingiu duração mínima
        if duration >= self.min_duration_sec:
            logger.info(
                "✅ [BUFFER] Buffer atingiu duração mínima, pronto para transcrição",
                meeting_id=meeting_id,
                participant_id=participant_id,
                track=track,
                duration_sec=round(duration, 2),
                chunks_count=len(buffer.audio_chunks)
            )
            return self._flush_buffer(key)
        
        # Verificar se atingiu duração máxima (forçar flush)
        if duration >= self.max_duration_sec:
            logger.info(
                "⏰ [BUFFER] Buffer atingiu duração máxima, forçando flush",
                meeting_id=meeting_id,
                participant_id=participant_id,
                track=track,
                duration_sec=round(duration, 2)
            )
            return self._flush_buffer(key)
        
        # Criar timer para flush automático após intervalo
        self.flush_timers[key] = asyncio.create_task(
            self._schedule_flush(key, self.flush_interval_sec)
        )
        
        return None
    
    async def _schedule_flush(self, key: Tuple[str, str, str], delay_sec: float):
        """Agenda flush automático após delay"""
        try:
            await asyncio.sleep(delay_sec)
            
            if key in self.buffers:
                buffer = self.buffers[key]
                duration = buffer.get_duration_sec()
                
                if duration > 0:
                    logger.info(
                        "⏰ [BUFFER] Flush automático por timeout",
                        meeting_id=key[0],
                        participant_id=key[1],
                        track=key[2],
                        duration_sec=round(duration, 2),
                        delay_sec=delay_sec
                    )
                    
                    if self.on_buffer_ready:
                        combined_wav = self._flush_buffer(key)
                        if combined_wav:
                            await self.on_buffer_ready(
                                meeting_id=key[0],
                                participant_id=key[1],
                                track=key[2],
                                wav_data=combined_wav,
                                sample_rate=buffer.sample_rate,
                                channels=buffer.channels,
                                timestamp=buffer.last_timestamp or int(time.time() * 1000)
                            )
        except asyncio.CancelledError:
            # Timer foi cancelado (novo chunk chegou)
            pass
    
    def _flush_buffer(self, key: Tuple[str, str, str]) -> Optional[bytes]:
        """Limpa buffer e retorna áudio combinado"""
        if key not in self.buffers:
            return None
        
        buffer = self.buffers[key]
        
        if not buffer.audio_chunks:
            return None
        
        # Combinar chunks em um único WAV
        combined_wav = buffer.get_combined_wav()
        
        # Limpar buffer
        buffer.clear()
        del self.buffers[key]
        
        # Cancelar timer se existir
        if key in self.flush_timers:
            self.flush_timers[key].cancel()
            del self.flush_timers[key]
        
        return combined_wav
    
    def clear_buffer(self, meeting_id: str, participant_id: str, track: str):
        """Limpa buffer específico"""
        key = (meeting_id, participant_id, track)
        if key in self.buffers:
            self.buffers[key].clear()
            del self.buffers[key]
        
        if key in self.flush_timers:
            self.flush_timers[key].cancel()
            del self.flush_timers[key]
    
    def clear_all(self):
        """Limpa todos os buffers"""
        for key in list(self.buffers.keys()):
            self._flush_buffer(key)
        
        for timer in self.flush_timers.values():
            timer.cancel()
        
        self.buffers.clear()
        self.flush_timers.clear()

