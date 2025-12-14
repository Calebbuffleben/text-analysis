# Text Analysis Service

Serviço Python para análise de transcrições em tempo real.

## Docker (Recomendado)

### Pré-requisitos

- Docker instalado
- Docker Compose instalado

### Usando Docker Compose (Recomendado)

```bash
# Na pasta apps/text-analysis
cd apps/text-analysis

# Iniciar o serviço
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar o serviço
docker-compose down
```

O serviço estará disponível em:
- **HTTP Health Check:** `http://localhost:8001/health`
- **Socket.IO:** `http://localhost:8001/socket.io/`

**Nota:** A porta 8001 no host está mapeada para a porta 8000 no container.

### Usando Docker diretamente

```bash
cd apps/text-analysis

# Construir a imagem
docker build -t text-analysis .

# Executar o container
docker run -d \
  --name live-meeting-text-analysis \
  -p 8001:8000 \
  -e LOG_LEVEL=INFO \
  -e SOCKETIO_CORS_ORIGINS=* \
  text-analysis

# Ver logs
docker logs -f live-meeting-text-analysis

# Parar o container
docker stop live-meeting-text-analysis
docker rm live-meeting-text-analysis
```

### Configurar Variáveis de Ambiente no Docker

Crie um arquivo `.env` na pasta `apps/text-analysis/` ou defina as variáveis diretamente no `docker-compose.yml`:

```bash
# Server
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO

# Socket.IO
SOCKETIO_CORS_ORIGINS=*

# ML Models
MODEL_CACHE_DIR=/app/models/.cache
MODEL_DEVICE=cpu
SENTIMENT_MODEL=neuralmind/bert-base-portuguese-cased
EMOTION_MODEL=cardiffnlp/twitter-roberta-base-emotion
ENABLE_ML_ANALYSIS=true

# Performance
ANALYSIS_BATCH_SIZE=1
ANALYSIS_MAX_LENGTH=512
```

### Verificar se Está Funcionando

```bash
# Testar health check
curl http://localhost:8001/health

# Deve retornar:
# {"status":"ok","service":"text-analysis"}
```

### Configurar Backend para Conectar

No arquivo `apps/backend/env`, configure:

```bash
# Text Analysis Service (rodando no Docker, porta 8001 no host)
TEXT_ANALYSIS_SERVICE_URL=http://localhost:8001
TEXT_ANALYSIS_ENABLED=true
TEXT_ANALYSIS_TIMEOUT_MS=5000
```

**Nota:** Quando o serviço está rodando no Docker, use `http://localhost:8001` (porta mapeada no host).

### Volumes Docker

O `docker-compose.yml` cria um volume para cache dos modelos ML:
- **Volume:** `text_analysis_models`
- **Localização no container:** `/app/models`
- Isso permite que os modelos baixados sejam persistidos entre reinicializações

---

## Rodar Localmente (Fora do Docker)

### Opção Rápida: Script de Setup Automático

```bash
cd apps/text-analysis
./scripts/setup-local.sh
```

O script automaticamente:
- Cria o ambiente virtual
- Instala todas as dependências
- Baixa o modelo spaCy em português

Depois, para executar:
```bash
source venv/bin/activate
python -m src.main
```

### Opção Manual: Setup Passo a Passo

#### 1. Instalar Dependências do Sistema

**macOS:**
```bash
# Verificar se já tem Xcode Command Line Tools
xcode-select --install

# Se necessário, instalar outras ferramentas via Homebrew
brew install python@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git
```

**Windows:**
- Instalar Python 3.11 do site oficial: https://www.python.org/downloads/
- Instalar Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/

### 2. Configurar Ambiente Virtual (Recomendado)

```bash
cd apps/text-analysis

# Criar ambiente virtual
python3.11 -m venv venv

# Ativar ambiente virtual
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 3. Instalar Dependências Python

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

### 4. Baixar Modelo spaCy em Português

```bash
python -m spacy download pt_core_news_sm
```

### 5. Configurar Variáveis de Ambiente (Opcional)

Crie um arquivo `.env` na pasta `apps/text-analysis/` (opcional, os valores padrão funcionam):

```bash
# Server
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO

# Socket.IO
SOCKETIO_CORS_ORIGINS=*

# ML Models
MODEL_CACHE_DIR=./models/.cache
MODEL_DEVICE=cpu
SENTIMENT_MODEL=neuralmind/bert-base-portuguese-cased
EMOTION_MODEL=cardiffnlp/twitter-roberta-base-emotion
ENABLE_ML_ANALYSIS=true

# Performance
ANALYSIS_BATCH_SIZE=1
ANALYSIS_MAX_LENGTH=512
```

**Nota:** Se usar `MODEL_CACHE_DIR` local (fora do Docker), ajuste o caminho para um diretório relativo ou absoluto que exista.

### 6. Executar o Serviço

```bash
# Certifique-se de estar na pasta apps/text-analysis
cd apps/text-analysis

# Ativar ambiente virtual (se ainda não estiver ativado)
source venv/bin/activate  # macOS/Linux
# ou
# venv\Scripts\activate  # Windows

# Executar o serviço
python -m src.main
```

O serviço estará disponível em:
- **HTTP Health Check:** `http://localhost:8000/health`
- **Socket.IO:** `http://localhost:8000/socket.io/`

### 7. Verificar se Está Funcionando

```bash
# Testar health check
curl http://localhost:8000/health

# Deve retornar:
# {"status":"ok","service":"text-analysis"}
```

### 8. Configurar Backend para Conectar

Quando rodando localmente (fora do Docker), o backend precisa apontar para a porta correta.

No arquivo `apps/backend/env`, configure:

```bash
# Text Analysis Service (serviço rodando localmente)
TEXT_ANALYSIS_SERVICE_URL=http://localhost:8000
TEXT_ANALYSIS_ENABLED=true
TEXT_ANALYSIS_TIMEOUT_MS=5000
```

**Nota:** Se o serviço estiver rodando na porta 8000 localmente, use `http://localhost:8000`. Se estiver em outra porta, ajuste conforme necessário.

---

## Troubleshooting

### Docker

**Ver logs do container:**
```bash
docker-compose logs -f text-analysis
```

**Reconstruir a imagem:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

**Container não inicia:**
- Verifique se a porta 8001 está disponível: `lsof -i :8001`
- Verifique os logs: `docker-compose logs text-analysis`
- Verifique o health check: `curl http://localhost:8001/health`

**Limpar volumes e recomeçar:**
```bash
docker-compose down -v
docker-compose up -d
```

### Local (Fora do Docker)

### Erro: "spacy model not found"
```bash
python -m spacy download pt_core_news_sm
```

### Erro: "ModuleNotFoundError"
Certifique-se de que:
1. O ambiente virtual está ativado
2. Todas as dependências foram instaladas: `pip install -r requirements.txt`
3. Está executando a partir da pasta correta: `apps/text-analysis`

### Erro ao instalar dependências (torch, transformers)
Algumas dependências ML podem ser grandes. Se houver problemas:
- Use uma conexão estável
- Aumente o timeout: `pip install --default-timeout=100 -r requirements.txt`

### Porta já em uso
Se a porta 8000 estiver em uso:
- Altere a variável `PORT` no arquivo `.env` ou
- Pare o serviço que está usando a porta 8000

