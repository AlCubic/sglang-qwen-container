#!/bin/bash
# Entrypoint скрипт для SGLang Docker контейнера (Dual-GPU версия)
# Инициализация и запуск SGLang сервера с поддержкой двух GPU

set -e

# Цветовые коды для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Логирование с timestamp
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [GPU2-INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [GPU2-WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [GPU2-ERROR]${NC} $1"
}

log_gpu() {
    echo -e "${MAGENTA}[$(date '+%Y-%m-%d %H:%M:%S')] [GPU2]${NC} $1"
}

# Проверка наличия GPU
check_gpu() {
    log_info "Проверка доступности GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "nvidia-smi найден, проверяем GPU..."
        
        echo ""
        log_gpu "Информация о GPU:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu,pci.bus_id --format=csv
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Подсчёт количества GPU
        local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        log_gpu "Обнаружено GPU: $gpu_count"
        
        if [ "$gpu_count" -ge 2 ]; then
            log_info "Доступно 2+ GPU, режим dual-GPU активирован"
            return 0
        else
            log_warn "Обнаружено менее 2 GPU. Доступно: $gpu_count"
            log_warn "Запуск в режиме single-GPU"
            return 1
        fi
    else
        log_warn "nvidia-smi не найден. GPU может быть недоступен."
        return 1
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
    
    log_gpu "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}"
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

# Проверка распределения модели по GPU
check_model_distribution() {
    log_gpu "Проверка распределения модели по GPU..."
    
    # Проверка использования памяти GPU
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | while read line; do
        log_gpu "GPU Memory: $line"
    done
}

# Очистка кэша
cleanup() {
    log_info "Очистка временных файлов..."
    
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find /root/.cache -type f -name "*.pyc" -delete 2>/dev/null || true
    find /logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Очистка кэша CUDA
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
}

# Основная функция запуска SGLang (Dual-GPU)
start_sglang_gpu2() {
    local model_path="${1:-$MODEL_PATH}"
    local host="${SGLANG_HOST:-0.0.0.0}"
    local port="${SGLANG_PORT:-5000}"
    local dtype="${SGLANG_DTYPE:-bfloat16}"
    
    log_info "=========================================="
    log_info "SGLang Dual-GPU Server Starting"
    log_info "=========================================="
    log_info "Модель: $model_path"
    log_info "Хост: $host"
    log_info "Порт: $port"
    log_info "Тип данных: $dtype"
    log_info "Режим: Dual-GPU (2x NVIDIA)"
    
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
    
    # Проверка GPU перед запуском
    check_model_distribution
    
    # Сборка аргументов для SGLang (Dual-GPU режим)
    SGLANG_ARGS=(
        "--model-path" "$model_path"
        "--host" "$host"
        "--port" "$port"
        "--dtype" "$dtype"
        "--trust-remote-code"
        "--log-level" "info"
        # Dual-GPU оптимизации
        "--tensor-parallel-size" "2"
        "--enable-flashinfer" "true"
    )
    
    log_info "Запуск SGLang Dual-GPU сервера..."
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
    log_info "SGLang Docker Container Started (Dual-GPU Mode)"
    log_info "Модель: Qwen2.5-0.5B-Instruct"
    log_info "Режим: Dual-GPU"
    log_info "=========================================="
    
    # Проверка GPU
    check_gpu
    
    # Проверка переменных окружения
    check_env
    
    # Очистка
    cleanup
    
    # Запуск SGLang
    MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-0.5B-Instruct}"
    start_sglang_gpu2 "$MODEL_PATH" &
    
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
