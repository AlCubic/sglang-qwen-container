#!/bin/bash
# Entrypoint скрипт для SGLang Docker контейнера (CPU Offload версия)
# Инициализация и запуск SGLang сервера с CPU offload для больших моделей

set -e

# Цветовые коды для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
ORANGE='\033[0;33m'
NC='\033[0m'

# Логирование с timestamp
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [OFFLOAD-INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [OFFLOAD-WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [OFFLOAD-ERROR]${NC} $1"
}

log_offload() {
    echo -e "${ORANGE}[$(date '+%Y-%m-%d %H:%M:%S')] [OFFLOAD]${NC} $1"
}

# Проверка GPU и CPU
check_resources() {
    log_info "Проверка доступных ресурсов..."
    
    # Проверка GPU
    if command -v nvidia-smi &> /dev/null; then
        log_offload "GPU информация:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
    else
        log_warn "nvidia-smi не найден, GPU может быть недоступен"
    fi
    
    # Проверка CPU
    log_offload "CPU информация:"
    if command -v nproc &> /dev/null; then
        log_offload "Доступных ядер: $(nproc)"
    fi
    
    if command -v free &> /dev/null; then
        log_offload "Оперативная память:"
        free -h | grep -E "Mem:|Swap:"
    fi
}

# Проверка переменных окружения
check_env() {
    log_info "Проверка переменных окружения..."
    
    MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-0.5B-Instruct}"
    log_info "Путь к модели: $MODEL_PATH"
    
    SGLANG_PORT="${SGLANG_PORT:-5000}"
    log_info "Порт сервера: $SGLANG_PORT"
    
    SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
    log_info "Тип данных: $SGLANG_DTYPE"
    
    # Offload специфичные переменные
    log_offload "CPU Offload: ${SGLANG_ENABLE_CPU_OFFLOAD:-true}"
    log_offload "FlashInfer workspace: ${FLASHINFER_WORKSPACE_DIR:-/tmp/flashinfer}"
}

# Проверка и загрузка модели
download_model() {
    local model_path="$1"
    local model_name="Qwen/Qwen2.5-0.5B-Instruct"
    
    if [ -d "$model_path" ]; then
        if [ -f "$model_path/config.json" ] && [ -f "$model_path/model.safetensors" ]; then
            log_info "Модель найдена локально: $model_path"
            return 0
        fi
    fi
    
    log_warn "Модель не найдена или неполная: $model_path"
    log_info "Начинаю загрузку модели с HuggingFace Hub..."
    
    mkdir -p "$model_path"
    
    if python3 -c "
from huggingface_hub import snapshot_download
import os

model_path = os.environ.get('MODEL_PATH', '/models/Qwen2.5-0.5B-Instruct')
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

print(f'Загрузка модели {model_name} в {model_path}...')
snapshot_download(
    repo_id=model_name,
    local_dir=model_path,
    local_dir_use_symlinks=False
)
print('Модель успешно загружена!')
"; then
        log_info "Модель успешно загружена: $model_path"
        return 0
    else
        log_error "Не удалось загрузить модель"
        return 1
    fi
}

# Настройка offload директории
setup_offload_dir() {
    local offload_dir="${OFFLOAD_DIR:-/tmp/offload}"
    
    log_offload "Настройка offload директории: $offload_dir"
    
    mkdir -p "$offload_dir"
    chmod 777 "$offload_dir"
    
    export SGLANG_CPU_OFFLOAD_DIR="$offload_dir"
    log_offload "SGLANG_CPU_OFFLOAD_DIR установлен: $offload_dir"
}

# Очистка кэша
cleanup() {
    log_info "Очистка временных файлов..."
    
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find /root/.cache -type f -name "*.pyc" -delete 2>/dev/null || true
    find /logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Очистка offload директории
    if [ -d "/tmp/offload" ]; then
        rm -rf /tmp/offload/* 2>/dev/null || true
    fi
}

# Основная функция запуска SGLang (CPU Offload)
start_sglang_offload() {
    local model_path="${1:-$MODEL_PATH}"
    local host="${SGLANG_HOST:-0.0.0.0}"
    local port="${SGLANG_PORT:-5000}"
    local dtype="${SGLANG_DTYPE:-bfloat16}"
    
    # Настройка offload
    setup_offload_dir
    
    log_info "=========================================="
    log_info "SGLang CPU Offload Server Starting"
    log_info "=========================================="
    log_info "Модель: $model_path"
    log_info "Хост: $host"
    log_info "Порт: $port"
    log_info "Тип данных: $dtype"
    log_info "Режим: CPU Offload (частичная загрузка в GPU)"
    
    # Проверка наличия локальной модели
    if [ -d "$model_path" ]; then
        if [ -f "$model_path/config.json" ] && [ -f "$model_path/model.safetensors" ]; then
            log_info "Найдена локальная модель: $model_path"
        else
            log_warn "Директория модели существует, но файлы неполные. Загрузка..."
            download_model "$model_path" || log_error "Не удалось загрузить модель"
        fi
    else
        log_info "Модель не найдена, начинаю загрузку..."
        download_model "$model_path" || log_error "Не удалось загрузить модель"
    fi
    
    # Сборка аргументов для SGLang (CPU Offload режим)
    SGLANG_ARGS=(
        "--model-path" "$model_path"
        "--host" "$host"
        "--port" "$port"
        "--dtype" "$dtype"
        "--trust-remote-code"
        "--log-level" "info"
        # CPU Offload параметры
        "--enable-cpu-offload" "true"
        "--cpu-offload-fraction" "0.5"
        "--enable-flashinfer" "true"
    )
    
    log_info "Запуск SGLang CPU Offload сервера..."
    log_info "Команда: python -m sglang.launch_server ${SGLANG_ARGS[*]}"
    
    exec python -m sglang.launch_server "${SGLANG_ARGS[@]}"
}

# Обработка сигналов
signal_handler() {
    log_info "Получен сигнал завершения..."
    kill -TERM $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    log_info "Сервер остановлен"
    exit 0
}

trap signal_handler SIGTERM SIGINT SIGHUP

# Основной скрипт
main() {
    log_info "=========================================="
    log_info "SGLang Docker Container Started (CPU Offload Mode)"
    log_info "Модель: Qwen2.5-0.5B-Instruct"
    log_info "Режим: CPU Offload"
    log_info "=========================================="
    
    # Проверка ресурсов
    check_resources
    
    # Проверка переменных окружения
    check_env
    
    # Очистка
    cleanup
    
    # Запуск SGLang
    MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-0.5B-Instruct}"
    start_sglang_offload "$MODEL_PATH" &
    
    PID=$!
    
    # Ожидание завершения
    wait $PID
    
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        log_error "SGLang сервер завершился с кодом: $exit_code"
    else
        log_info "SGLang сервер успешно остановлен"
    fi
    
    exit $exit_code
}

main "$@"
