#!/bin/bash
# Entrypoint скрипт для SGLang Docker контейнера
# Инициализация и запуск SGLang сервера с Qwen2.5-0.5B-Instruct

set -e

# Цветовые коды для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Логирование с timestamp
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG]${NC} $1"
}

# Проверка наличия GPU
check_gpu() {
    log_info "Проверка доступности GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "nvidia-smi найден, проверяем GPU..."
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        return 0
    else
        log_warn "nvidia-smi не найден. GPU может быть недоступен внутри контейнера."
        return 1
    fi
}

# Проверка переменных окружения
check_env() {
    log_info "Проверка переменных окружения..."
    
    # Проверка модели
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
    log_info "Путь к модели: $MODEL_PATH"
    
    # Проверка HuggingFace токена (опционально)
    if [ -n "$HF_TOKEN" ]; then
        log_info "HuggingFace токен обнаружен"
    else
        log_warn "HuggingFace токен не установлен. Модель будет загружена публично."
    fi
    
    # Проверка dtype
    SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
    log_info "Тип данных: $SGLANG_DTYPE"
}

# Очистка кэша и временных файлов
cleanup() {
    log_info "Очистка временных файлов..."
    
    # Очистка Python кэша
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find /root/.cache -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Очистка логов старше 7 дней
    find /logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
}

# Основная функция запуска SGLang
start_sglang() {
    local model_path="${1:-$MODEL_PATH}"
    local host="${SGLANG_HOST:-0.0.0.0}"
    local port="${SGLANG_PORT:-30000}"
    local dtype="${SGLANG_DTYPE:-bfloat16}"
    
    log_info "Запуск SGLang сервера..."
    log_info "Модель: $model_path"
    log_info "Хост: $host"
    log_info "Порт: $port"
    log_info "Тип данных: $dtype"
    
    # Проверка и создание директории для кэша модели
    mkdir -p /models
    mkdir -p /logs
    
    # Сборка аргументов для SGLang
    SGLANG_ARGS=(
        "--model-path" "$model_path"
        "--host" "$host"
        "--port" "$port"
        "--dtype" "$dtype"
        "--trust-remote-code"
        "--enforce-eager"
        "--log-level" "info"
        "--log-requests"
    )
    
    # Добавление опциональных параметров из конфигурации
    if [ -f "/app/config/custom_sglang.json" ]; then
        log_info "Загрузка пользовательской конфигурации..."
        # Параметры могут быть добавлены здесь при необходимости
    fi
    
    # Добавление параметров для оптимизации памяти на RTX 4000
    local gpu_memory_fraction="${GPU_MEMORY_FRACTION:-0.85}"
    SGLANG_ARGS+=("--gpu-memory-utilization" "$gpu_memory_fraction")
    
    # Добавление chunked prefill для лучшей производительности
    if [ "${CHUNKED_PREFILL:-true}" = "true" ]; then
        SGLANG_ARGS+=("--enable-chunked-prefill")
    fi
    
    # Добавление max concurrent tokens
    local max_concurrent="${MAX_CONCURRENT_TOKENS:-256}"
    SGLANG_ARGS+=("--max-concurrent-tokens" "$max_concurrent")
    
    log_debug "Полные аргументы SGLang: ${SGLANG_ARGS[*]}"
    
    # Запуск SGLang сервера
    log_info "Запуск команды: python -m sglang.launch_server ${SGLANG_ARGS[*]}"
    
    exec python -m sglang.launch_server "${SGLANG_ARGS[@]}"
}

# Функция обработки сигналов graceful shutdown
signal_handler() {
    log_info "Получен сигнал завершения, останавливаю сервер..."
    kill -TERM $PID
    wait $PID 2>/dev/null || true
    log_info "Сервер остановлен"
    exit 0
}

# Trap сигналы
trap signal_handler SIGTERM SIGINT SIGHUP

# Основной скрипт
main() {
    log_info "=========================================="
    log_info "SGLang Docker Container Started"
    log_info "Модель: Qwen2.5-0.5B-Instruct"
    log_info "GPU: Quadro RTX 4000 (Turing)"
    log_info "=========================================="
    
    # Проверка GPU
    check_gpu || log_warn "GPU не обнаружен, продолжаем..."
    
    # Проверка переменных окружения
    check_env
    
    # Очистка
    cleanup
    
    # Запуск SGLang
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
    start_sglang "$MODEL_PATH" &
    
    # Сохранение PID для graceful shutdown
    PID=$!
    
    # Ожидание завершения процесса
    wait $PID
    
    # Код завершения
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        log_error "SGLang сервер завершился с кодом: $exit_code"
    else
        log_info "SGLang сервер успешно остановлен"
    fi
    
    exit $exit_code
}

# Запуск main функции
main "$@"
