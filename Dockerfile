# SGLang Docker Container с поддержкой GPU для Qwen2.5-0.5B-Instruct
# Использует официальный образ SGLang с предустановленными зависимостями

FROM lmsysorg/sglang:v0.4.8.post1-cu126

LABEL maintainer="developer@example.com"
LABEL description="SGLang Inference Server for Qwen2.5-0.5B-Instruct"
LABEL version="1.0.0"

# Переменные окружения для оптимизации производительности
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer
ENV TORCH_CUDA_ARCH_LIST=7.5

# Создание рабочей директории
WORKDIR /app

# Копирование конфигурационных файлов
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Создание директории для моделей, логов и Flashinfer workspace
# Создаём ДО переключения на пользователя для избежания проблем с правами
RUN mkdir -p /models /data /logs /tmp/flashinfer && chmod 777 /tmp/flashinfer

# Порт для SGLang API
EXPOSE 5000

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
