FROM python:3.11-slim

# Metadados
LABEL maintainer="live-meeting-team"
LABEL description="Text Analysis Service with BERT for Portuguese"
LABEL version="1.0.0"

# Instalar dependências do sistema
# faster-whisper precisa de:
# - ffmpeg (binário do sistema) para processar áudio
# - build-essential para compilar ctranslate2 (dependência do faster-whisper)
# - libffi-dev e libssl-dev para compilar algumas dependências Python
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro (para melhor cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Baixar recursos NLTK durante build (acelera primeiro uso)
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt_tab', quiet=True)"

# Criar diretório para modelos com permissões corretas
RUN mkdir -p /app/models && \
    chmod 755 /app/models

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models/.cache
ENV HF_HOME=/app/models/.cache
ENV NLTK_DATA=/app/models/nltk_data

# Copiar código (último passo para melhor cache)
COPY src/ ./src/

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s \
           --timeout=10s \
           --start-period=30s \
           --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização
CMD ["python", "-m", "src.main"]

