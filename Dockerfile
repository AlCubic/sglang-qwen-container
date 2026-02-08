# SGLang Docker Container с поддержкой GPU для Qwen2.5-0.5B-Instruct
# Базируется на официальном CUDA образе с Python 3.11

FROM nvidia/cuda:12.6.1-base-ubuntu22.04

LABEL maintainer="developer@example.com"
LABEL description="SGLang Inference Server for Qwen2.5-0.5B-Instruct"
LABEL version="1.0.0"

# Переменные окружения для оптимизации производительности
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    net-tools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Установка Python 3.11 и зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создание виртуального окружения Python
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка SGLang из PyPI (стабильная версия v0.4.x)
RUN pip install --upgrade pip setuptools wheel
RUN pip install sglang==0.4.8.post5 \
    --no-cache-dir

# Установка дополнительных зависимостей для мониторинга и аналитики
RUN pip install prometheus-client \
    fastapi \
    uvicorn \
    --no-cache-dir

# Создание рабочей директории
WORKDIR /app

# Копирование конфигурационных файлов
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Создание директории для моделей
RUN mkdir -p /models /data /logs

# Скачивание модели Qwen2.5-0.5B-Instruct при первом запуске
# Модель будет загружена автоматически при старте сервера

# Порт для SGLang API
EXPOSE 30000

# Скрипт инициализации
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Переключение на непривилегированного пользователя для безопасности
RUN groupadd -r sglang && useradd -r -g sglang sglang
RUN chown -R sglang:sglang /app /models /data /logs
USER sglang

# Entrypoint скрипт
ENTRYPOINT ["/app/entrypoint.sh"]

# Команда по умолчанию запускает SGLang сервер
CMD ["--model-path", "Qwen/Qwen2.5-0.5B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "30000", \
     "--dtype", "bfloat16", \
     "--trust-remote-code", \
     "--enforce-eager"]
