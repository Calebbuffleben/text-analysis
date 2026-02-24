# Arquitetura do buffer de áudio (modo contínuo)

Explicação simplificada do fluxo de áudio com **buffer circular** e **worker de janela deslizante**.

---

## Visão geral

No modo contínuo (`CONTINUOUS_WORKER_ENABLED=true`), cada stream de áudio (reunião + participante + track) tem:

- **Um buffer** (ring): guarda os últimos X segundos de áudio.
- **Um worker**: lê janelas desse buffer e manda para transcrição via callback.

Quem junta os dois é o stream **ContinuousAudioStreamManager**: recebe os chunks, guarda no buffer certo e garante um worker por stream.

---

## O Buffer (ring)

**Arquivo:** `src/services/audio_buffer/circular_buffer.py`  
**Classe:** `CircularAudioBuffer`

- **O que faz:** guarda áudio em um **buffer circular** (ring) de tamanho fixo.
- **Entrada:** chunks de áudio entram por `append_wav` / `append_pcm`.
- **Comportamento:** quando o buffer enche, **novos dados sobrescrevem** os mais antigos (o “anel” dá a volta). O buffer **sobrescreve**; ninguém apaga manualmente.
- **Leitura:** o buffer também permite **consultar** sem consumir: `has_enough_audio(segundos)`, `get_written_samples_total()`, `snapshot_last_samples(n)` — este último devolve uma **cópia** das últimas N amostras; o conteúdo do ring não é alterado pela leitura.

Em resumo: **coloca o áudio no buffer**; quando está cheio, **ele mesmo sobrescreve** o que é mais antigo.

---

## O Worker

**Arquivo:** `src/services/audio_buffer/sliding_worker.py`  
**Classe:** `SlidingWindowWorker`

- **O que faz:** em loop, **lê** janelas de áudio do buffer (do ring) e envia cada janela para o **callback** injetado (ex.: `on_buffer_ready` no `socketio_server`), que dispara transcrição e análise.
- **Comportamento:** só **lê** (copia a janela). **Não apaga** nem altera o buffer; quem sobrescreve é o buffer quando chegam novos chunks.
- **Janela:** tamanho fixo (ex.: 5 s). A cada passo o worker pega as **últimas** N amostras do ring — a janela “desliza” no tempo (sempre os últimos X segundos naquele momento).

Em resumo: **lê** do buffer e **entrega** o WAV da janela ao callback; o callback é quem chama a transcrição.

---

## Onde está cada um

| Papel   | Onde está |
|--------|-----------|
| Buffer | `src/services/audio_buffer/circular_buffer.py` → `CircularAudioBuffer` |
| Worker | `src/services/audio_buffer/sliding_worker.py` → `SlidingWindowWorker` |
| Quem liga os dois | `src/services/audio_buffer/stream_manager.py` → `ContinuousAudioStreamManager` |

---

## Janelas deslizantes (resumo)

- **Janela** = um bloco de tamanho fixo (ex.: 5 s).
- **Deslizante** = a cada passo você olha de novo “os últimos X segundos”; esse intervalo vai “andando” no tempo (entra áudio novo, sai o mais antigo da janela).
- A janela **não** dura o tempo total do áudio; ela tem **duração fixa** (ex.: 5 s). O que desliza é **qual** trecho de 5 s está sendo lido.

---

## Quem lê e quem sobrescreve

- **Worker:** só **lê** (copia a janela). Não apaga nem sobrescreve o buffer.
- **Buffer:** quando chega áudio novo e já está cheio, a **própria escrita** no ring **sobrescreve** o trecho mais antigo.

## Fluxo em uma frase

Chunks de áudio → **manager** coloca no **buffer** do stream → **worker** lê janelas do buffer e chama o **callback** → callback dispara transcrição e análise. O buffer sobrescreve quando enche; o worker só lê.
