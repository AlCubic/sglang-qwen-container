# SGLang + Qwen2.5-0.5B-Instruct Docker Deployment

Данный проект содержит полную конфигурацию для развёртывания высокопроизводительного LLM-сервера на базе фреймворка SGLang с моделью Qwen2.5-0.5B-Instruct. Поддерживаются различные режимы развёртывания: single-GPU, dual-GPU, CPU-only и CPU Offload.

## Содержание

1. [Обзор решения](#обзор-решения)
2. [Режимы развёртывания](#режимы-развёртывания)
3. [Системные требования](#системные-требования)
4. [Установка и настройка](#установка-и-настройка)
5. [Запуск и эксплуатация](#запуск-и-эксплуатация)
6. [Конфигурация](#конфигурация)
7. [Мониторинг и обслуживание](#мониторинг-и-обслуживание)
8. [Производительность](#производительность)
9. [Устранение неполадок](#устранение-неполадок)

## Обзор решения

### Архитектура решения

Проект представляет собой контейнеризированное решение для инференса языковых моделей, включающее следующие компоненты:

- **SGLang v0.4.8.post1** — высокопроизводительный фреймворк для обслуживания больших языковых моделей
- **Qwen2.5-0.5B-Instruct** — компактная языковая модель от Alibaba с 500 миллионами параметров
- **Docker Container** — контейнер на базе официального образа lmsysorg/sglang с CUDA 12.6

### Ключевые характеристики модели

| Параметр | Значение |
|----------|----------|
| Количество параметров | 500 млн |
| Контекстное окно | 32768 токенов |
| Требования к VRAM (BF16) | ~2 ГБ |
| Поддержка квантизации | Да (GPTQ, AWQ, GGUF) |
| Языки | Многоязычная (включая русский) |

## Режимы развёртывания

### 1. Single-GPU (по умолчанию)

Запуск на одном GPU. Подходит для большинства случаев использования.

**Требования:**
- 1 GPU с минимум 8 ГБ VRAM
- CUDA 12.0+

**Запуск:**
```bash
./scripts/start.sh build
./scripts/start.sh start
```

### 2. Dual-GPU

Запуск на двух GPU с использованием tensor parallelism для увеличения производительности.

**Требования:**
- 2 GPU с минимум 8 ГБ VRAM каждый
- CUDA 12.0+

**Запуск:**
```bash
DEPLOYMENT_MODE=dual-gpu ./scripts/start.sh build
DEPLOYMENT_MODE=dual-gpu ./scripts/start.sh start
```

**Преимущества:**
- Увеличенная производительность
- Возможность использования более крупных моделей
- Балансировка нагрузки между GPU

### 3. CPU-Only

Запуск без GPU, используя только CPU для инференса. Подходит для тестирования или систем без GPU.

**Требования:**
- CPU с поддержкой AVX2
- Минимум 16 ГБ оперативной памяти
- 8+ ядер CPU рекомендуется

**Запуск:**
```bash
DEPLOYMENT_MODE=cpu-only ./scripts/start.sh build
DEPLOYMENT_MODE=cpu-only ./scripts/start.sh start
```

**Оптимизация:**
```bash
export SGLANG_CPU_OMP_NUM_THREADS=16  # Количество потоков
```

**Ожидаемая производительность:**
- TTFT: 500-2000 мс
- TPS: 5-20 токенов/с

### 4. CPU Offload

Гибридный режим, где часть модели загружается в GPU, а остальная часть хранится в CPU и подгружается по мере необходимости. Подходит для больших моделей, которые не помещаются полностью в видеопамять.

**Требования:**
- 1 GPU с минимум 4 ГБ VRAM
- CPU с поддержкой AVX2
- Минимум 32 ГБ оперативной памяти

**Запуск:**
```bash
DEPLOYMENT_MODE=offload ./scripts/start.sh build
DEPLOYMENT_MODE=offload ./scripts/start.sh start
```

**Конфигурация:**
```bash
export SGLANG_CPU_OFFLOAD_FRACTION=0.5  # Доля модели в CPU (0.0-1.0)
```

**Преимущества:**
- Возможность запуска больших моделей
- Экономия VRAM
- Компромисс между производительностью и требованиями к памяти

## Системные требования

### Аппаратные требования

| Компонент | Single-GPU | Dual-GPU | CPU-Only | Offload |
|-----------|------------|----------|----------|---------|
| GPU | NVIDIA RTX 4060+ 8GB | 2x NVIDIA RTX 4060+ 8GB | Нет | NVIDIA RTX 4060+ 4GB |
| VRAM | 8+ ГБ | 16+ ГБ | Нет | 4+ ГБ |
| Системная память | 16 ГБ | 32 ГБ | 16+ ГБ | 32+ ГБ |
| CPU | 4 ядра | 8 ядер | 8+ ядер | 8 ядер |
| Диск | 50 ГБ SSD | 100 ГБ NVMe | 50 ГБ SSD | 100 ГБ NVMe |

### Программные требования

| Компонент | Требуемая версия |
|-----------|------------------|
| NVIDIA Driver | 470.x или выше |
| Docker | 20.10 или выше |
| Docker Compose | 2.0+ |
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

**Выбор режима:**

```bash
# Single-GPU (по умолчанию)
./scripts/start.sh build
./scripts/start.sh start

# Dual-GPU
DEPLOYMENT_MODE=dual-gpu ./scripts/start.sh build
DEPLOYMENT_MODE=dual-gpu ./scripts/start.sh start

# CPU-Only
DEPLOYMENT_MODE=cpu-only ./scripts/start.sh build
DEPLOYMENT_MODE=cpu-only ./scripts/start.sh start

# CPU Offload
DEPLOYMENT_MODE=offload ./scripts/start.sh build
DEPLOYMENT_MODE=offload ./scripts/start.sh start
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

Основные параметры настраиваются через переменные окружения:

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| MODEL_PATH | /models/Qwen2.5-0.5B-Instruct | Локальный путь к модели |
| SGLANG_HOST | 0.0.0.0 | IP для прослушивания |
| SGLANG_PORT | 5000 | Порт сервера |
| SGLANG_DTYPE | bfloat16 | Тип данных (float32 для CPU) |
| SGLANG_CPU_OMP_NUM_THREADS | 8 | Количество потоков CPU |
| SGLANG_ENABLE_CPU_OFFLOAD | false | Включение CPU offload |

### Переменные окружения Docker Compose

```yaml
environment:
  - HOME=/tmp
  - SGLANG_PORT=5000
  - SGLANG_DTYPE=bfloat16
  - CUDA_VISIBLE_DEVICES=0
  - FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer
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
| `/v1/models` | GET | Список моделей |

## Производительность

### Ожидаемые показатели

| Режим | TTFT | TPS | VRAM |
|-------|------|-----|------|
| Single-GPU | 15-30 мс | 50-80 | 3-4 ГБ |
| Dual-GPU | 10-20 мс | 100-150 | 6-8 ГБ |
| CPU-Only | 500-2000 мс | 5-20 | 0 ГБ |
| Offload | 50-100 мс | 30-50 | 2-3 ГБ |

## Устранение неполадок

### Ошибка: unrecognized arguments

**Проблема**: SGLang v0.4.8 не поддерживает некоторые устаревшие параметры.

**Решение**: Используйте только поддерживаемые параметры, как указано в соответствующих entrypoint скриптах.

### Ошибка: CUDA out of memory

**Проблема**: Недостаточно памяти GPU.

**Решение**: 
- Уменьшите количество одновременных запросов
- Переключитесь на режим CPU Offload
- Используйте CPU-Only режим для тестирования

### Ошибка: PermissionError

**Проблема**: Нет прав на запись в директорию.

**Решение**:
```bash
sudo chmod -R 777 models data logs
```

### Ошибка: GPU not found

**Проблема**: GPU недоступен в контейнере.

**Решение**:
- Проверьте установку NVIDIA Container Toolkit
- Проверьте драйверы NVIDIA
- Для CPU-Only режима установите DEPLOYMENT_MODE=cpu-only

### Ошибка: ModuleNotFoundError

**Проблема**: Отсутствуют Python зависимости.

**Решение**: Используйте официальный образ SGLang, который включает все зависимости.

## Структура проекта

```
sglang-qwen-container/
├── Dockerfile                    # Single-GPU конфигурация
├── Dockerfile.cpu                # CPU-only конфигурация
├── Dockerfile.gpu2               # Dual-GPU конфигурация
├── Dockerfile.offload            # CPU Offload конфигурация
├── docker-compose.yml           # Оркестрация контейнеров
├── README.md                    # Документация
├── .gitignore                   # Исключения для Git
├── requirements.txt             # Python зависимости
│
├── config/
│   └── custom_sglang.json       # Конфигурация SGLang
│
├── scripts/
│   ├── entrypoint.sh            # Entrypoint (single-GPU)
│   ├── entrypoint_cpu.sh        # Entrypoint (CPU-only)
│   ├── entrypoint_gpu2.sh       # Entrypoint (dual-GPU)
│   ├── entrypoint_offload.sh    # Entrypoint (offload)
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
