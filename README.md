# SGLang + Qwen2.5-0.5B-Instruct Docker Deployment

Данный проект содержит полную конфигурацию для развёртывания высокопроизводительного LLM-сервера на базе фреймворка SGLang с моделью Qwen2.5-0.5B-Instruct, оптимизированного для работы на GPU NVIDIA Quadro RTX 4000 (архитектура Turing).

## Содержание

1. [Обзор решения](#обзор-решения)
2. [Системные требования](#системные-требования)
3. [Установка и настройка](#установка-и-настройка)
4. [Запуск и эксплуатация](#запуск-и-эксплуатация)
5. [Конфигурация](#конфигурация)
6. [Мониторинг и обслуживание](#мониторинг-и-обслуживание)
7. [Производительность](#производительность)
8. [Устранение неполадок](#устранение-неполадок)
9. [Важные исправления и особенности](#важные-исправления-и-особенности)

## Обзор решения

### Архитектура решения

Проект представляет собой контейнеризированное решение для инференса языковых моделей, включающее следующие компоненты:

- **SGLang v0.4.8.post1** — высокопроизводительный фреймворк для обслуживания больших языковых моделей
- **Qwen2.5-0.5B-Instruct** — компактная языковая модель от Alibaba с 500 миллионами параметров
- **Docker Container** — контейнер на базе официального образа lmsysorg/sglang:v0.4.8.post1-cu126 с CUDA 12.6

### Ключевые характеристики модели

| Параметр | Значение |
|----------|----------|
| Количество параметров | 500 млн |
| Контекстное окно | 32768 токенов |
| Требования к VRAM (BF16) | ~2 ГБ |
| Поддержка квантизации | Да (GPTQ, AWQ, GGUF) |
| Языки | Многоязычная (включая русский) |

## Системные требования

### Аппаратные требования

| Компонент | Минимальные требования | Рекомендуемые |
|-----------|------------------------|---------------|
| GPU | NVIDIA Quadro RTX 4000 8GB | NVIDIA RTX 4060+ |
| VRAM | 8 ГБ | 8+ ГБ |
| Системная память | 16 ГБ | 32 ГБ |
| CPU | 4 ядра | 8 ядер |
| Диск | 50 ГБ SSD | 100+ ГБ NVMe SSD |

Quadro RTX 4000 построена на архитектуре Turing (CUDA capability 7.5) с 8 ГБ GDDR6 памяти.

### Программные требования

| Компонент | Требуемая версия |
|-----------|------------------|
| NVIDIA Driver | 470.x или выше |
| Docker | 20.10 или выше |
| Docker Compose | 2.0+ (используйте `docker compose` без дефиса) |
| NVIDIA Container Toolkit | 1.13 или выше |
| CUDA Toolkit | 12.0+ |

### Проверка GPU

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu22.04 nvidia-smi
```

## Установка и настройка

### Предварительная загрузка модели (рекомендуется)

Для ускорения первого запуска рекомендуется предварительно загрузить модель на сервер:

```bash
# Установка зависимостей
pip3 install --break-system-packages --user huggingface_hub

# Создание директории для модели
mkdir -p ~/sglang-qwen-container/models

# Загрузка модели
cd ~/sglang-qwen-container/models
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='Qwen2.5-0.5B-Instruct')"
```

### Подготовка окружения

```bash
# Клонирование репозитория
cd ~/sglang-qwen-project

# Установка прав на исполнение скриптов
chmod +x scripts/*.sh

# Создание необходимых директорий
mkdir -p models data logs

# Проверка прав на директорию models
sudo chown -R $USER:$USER models data logs
```

### Сборка и запуск

```bash
# Сборка Docker образа
cd ~/sglang-qwen-container
docker compose build

# Запуск контейнера
docker compose up -d

# Проверка статуса
docker ps | grep sglang
```

## Запуск и эксплуатация

### Базовые операции

```bash
# Запуск
docker compose up -d

# Просмотр логов
docker logs sglang-qwen-inference -f

# Перезапуск
docker compose restart

# Остановка
docker compose down
```

### Тестирование API

```bash
# Проверка информации о модели
curl http://localhost:5000/get_model_info

# Генерация текста
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Какой столица Франции?", "max_tokens": 100}'

# Проверка здоровья
curl http://localhost:5000/health
```

## Конфигурация

### Параметры SGLang

Основные параметры настраиваются через переменные окружения в docker-compose.yml:

| Параметр | Значение | Описание |
|----------|----------|----------|
| MODEL_PATH | /models/Qwen2.5-0.5B-Instruct | Локальный путь к модели |
| SGLANG_HOST | 0.0.0.0 | IP для прослушивания |
| SGLANG_PORT | 5000 | Порт сервера |
| SGLANG_DTYPE | bfloat16 | Тип данных |
| HOME | /tmp | Для избежания ошибок flashinfer |
| FLASHINFER_WORKSPACE_DIR | /tmp/flashinfer | Рабочая директория flashinfer |
| TORCH_CUDA_ARCH_LIST | 7.5 | CUDA architecture для Turing |

### Оптимизация для RTX 4000

Для Quadro RTX 4000 критически важны следующие настройки:

```yaml
environment:
  - HOME=/tmp
  - FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer
  - TORCH_CUDA_ARCH_LIST=7.5
```

## Мониторинг

### Проверка состояния

```bash
# Статус контейнера
docker ps | grep sglang

# Использование GPU
nvidia-smi

# Логи контейнера
docker logs sglang-qwen-inference --tail 50
```

### API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/generate` | POST | Генерация текста |
| `/health` | GET | Проверка работоспособности |
| `/get_model_info` | GET | Информация о модели |

## Производительность

### Ожидаемые показатели на RTX 4000

| Метрика | Значение |
|---------|----------|
| TTFT | 15-30 мс |
| TPS | 50-80 токенов/с |
| Latency (100 токенов) | 1.5-2.5 с |
| VRAM usage | 3-4 ГБ |

## Устранение неполадок

### Ошибка: unrecognized arguments: --enforce-eager

**Проблема**: SGLang v0.4.8 не поддерживает некоторые устаревшие параметры.

**Решение**: Используйте только поддерживаемые параметры:

```bash
--model-path /models/Qwen2.5-0.5B-Instruct
--host 0.0.0.0
--port 5000
--dtype bfloat16
--trust-remote-code
--log-level info
```

### Ошибка: PermissionError: '/home/sglang'

**Проблема**: flashinfer пытается создать workspace в домашней директории.

**Решение**: Установите HOME=/tmp в docker-compose.yml:

```yaml
environment:
  - HOME=/tmp
```

### Ошибка: CUDA out of memory

**Проблема**: Недостаточно памяти GPU.

**Решение**: Уменьшите количество одновременных запросов или перезапустите контейнер.

### Ошибка: ModuleNotFoundError

**Проблема**: Отсутствуют Python зависимости.

**Решение**: Используйте официальный образ SGLang, который включает все зависимости:

```dockerfile
FROM lmsysorg/sglang:v0.4.8.post1-cu126
```

## Важные исправления и особенности

### Проблемы при разработке и их решения

В процессе развёртывания были решены следующие критические проблемы:

#### 1. Официальный образ vs Самосборный

**Проблема**: При попытке собрать образ с нуля возникали ошибки отсутствия зависимостей (orjson, uvloop, psutil, torch).

**Решение**: Использование официального образа lmsysorg/sglang:v0.4.8.post1-cu126, который включает все необходимые зависимости.

#### 2. Flashinfer Permission Error

**Проблема**: `PermissionError: [Errno 13] Permission denied: '/home/sglang'` при запуске контейнера.

**Решение**: Комбинация двух подходов:
- Создание `/tmp/flashinfer` с правами 777 в Dockerfile перед переключением пользователя
- Установка `HOME=/tmp` в docker-compose.yml

#### 3. Устаревшие аргументы SGLang

**Проблема**: `unrecognized arguments: --enforce-eager --gpu-memory-utilization --enable-chunked-prefill --max-concurrent-tokens`

**Решение**: Удаление устаревших параметров из entrypoint.sh. SGLang v0.4.8 автоматически оптимизирует эти параметры.

#### 4. CUDA Architecture для Turing

**Проблема**: Медленная загрузка модели или ошибки совместимости.

**Решение**: Установка `TORCH_CUDA_ARCH_LIST=7.5` для оптимизации под архитектуру Turing.

#### 5. Pre-download модели

**Проблема**: Долгая загрузка модели при первом запуске контейнера.

**Решение**: Предварительная загрузка модели на сервер через huggingface_hub с последующим монтированием локальной директории.

### Команды для быстрого развёртывания

```bash
# 1. Клонирование и подготовка
cd ~/sglang-qwen-container
git pull origin master

# 2. Загрузка модели (опционально)
mkdir -p models && cd models
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='Qwen2.5-0.5B-Instruct')"
cd ..

# 3. Сборка и запуск
docker compose build --no-cache
docker compose up -d

# 4. Проверка
docker logs sglang-qwen-inference -f
```

### Структура проекта

```
sglang-qwen-container/
├── Dockerfile                    # Docker конфигурация
├── docker-compose.yml           # Оркестрация контейнеров
├── README.md                    # Документация
├── .gitignore                   # Исключения для Git
├── requirements.txt             # Python зависимости
│
├── config/
│   └── custom_sglang.json       # Конфигурация SGLang
│
├── scripts/
│   ├── entrypoint.sh            # Entrypoint контейнера
│   └── start.sh                 # Скрипт управления
│
├── models/                      # Локальные модели (исключено из Git)
├── data/                        # Рабочие данные
└── logs/                        # Логи
```

## Ссылки

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Qwen2.5 Documentation](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
