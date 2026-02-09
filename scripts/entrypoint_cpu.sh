#!/bin/bash
# Entrypoint скрипт для SGLang Docker контейнера (CPU-only версия)
# Инициализация и запуск SGLang сервера с Qwen2.5-0.5B-Instruct на CPU

set -e

# Цветовые коды для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Логирование с timestamp
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [CPU-INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [CPU-WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [CPU-ERROR]${NC} $1"
}

log_cpu() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] [CPU]${NC} $1"
}

# Проверка CPU
check_cpu() {
    log_info "Проверка CPU..."
    
    log_cpu "CPU информация:"
    if command -v nproc &> /dev/null; then
        log_cpu "Доступных ядер: $(nproc)"
    fi
    
    if command -v lscpu &> /dev/null; then
        lscpu | grep -E "Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|CPU MHz" | head -5
    fi
    
    # Проверка памяти
    log_cpu "Оперативная память:"
    if command -v free &> /dev/null; then
        free -h
    fi
    
    return 0
}

# Оптимизация CPU для инференса
optimize_cpu() {
    log_info "Оптимизация CPU для инференса..."
    
    # Установка количества потоков
    local num_threads="${SGLANG_CPU_OMP_NUM_THREADS:-8}"
    
    if [ -n "$num_threads" ]; then
        export OMP_NUM_THREADS="$num_threads"
        export MKL_NUM_THREADS="$num_threads"
        export NUMEXPR_NUM_THREADS="$num_threads"
        export OPENBLAS_NUM_THREADS="$num_threads"
        
        log_info "Количество потоков: $num_threads"
    fi
    
    # Отключение турбо-буста для стабильной производительности
    if [ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    fi
}

# Проверка переменных окружения
check_env() {
    log_info "Проверка переменных окружения..."
    
    MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-0.5B-Instruct}"
    log_info "Путь к модели: $MODEL_PATH"
    
    SGLANG_PORT="${SGLANG_PORT:-5000}"
    log_info "Порт сервера: $SGLANG_PORT"
    
    SGLANG_DTYPE="${SGLANG_DTYPE:-float32}"
    log_info "Тип данных (CPU): $SGLANG_DTYPE"
    
    # CPU-only специфичные переменные
    log_cpu "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-не установлено}"
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

# Очистка кэша
cleanup() {
    log_info "Очистка временных файлов..."
    
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find /root/.cache -type f -name "*.pyc" -delete 2>/dev/null || true
    find /logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
}

# Основная функция запуска SGLang (CPU-only)
start_sglang_cpu() {
    local model_path="${1:-$MODEL_PATH}"
    local host="${SGLANG_HOST:-0.0.0.0}"
    local port="${SGLANG_PORT:-5000}"
    local dtype="${SGLANG_DTYPE:-float32}"
    
    log_info "=========================================="
    log_info "SGLang CPU-Only Server Starting"
    log_info "=========================================="
    log_info "Модель: $model_path"
    log_info "Хост: $host"
    log_info "Порт: $port"
    log_info "Тип данных: $dtype"
    log_info "Режим: CPU-only (без GPU)"
    
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
    
    # Сборка аргументов для SGLang (CPU режим)
    SGLANG_ARGS=(
        "--model-path" "$model_path"
        "--host" "$host"
        "--port" "$port"
        "--dtype" "$dtype"
        "--trust-remote-code"
        "--log-level" "info"
        # CPU-only параметры
        "--no-avx" "false"
        "--disable-flashinfer" "true"
    )
    
    log_info "Запуск SGLang CPU сервера..."
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
    log_info "SGLang Docker Container Started (CPU Mode)"
    log_info "Модель: Qwen2.5-0.5B-Instruct"
    log_info "Режим: CPU-only"
    log_info "=========================================="
    
    # Проверка CPU
    check_cpu
    
    # Оптимизация CPU
    optimize_cpu
    
    # Проверка переменных окружения
    check_env
    
    # Очистка
    cleanup
    
    # Запуск SGLang
    MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-0.5B-Instruct}"
    start_sglang_cpu "$MODEL_PATH" &
    
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
