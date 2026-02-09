#!/bin/bash
# Скрипт запуска SGLang Docker контейнера с Qwen2.5-0.5B-Instruct
# Поддержка: Single-GPU, Dual-GPU, CPU-only, CPU Offload

set -e

# Цветовые коды для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
ORANGE='\033[0;33m'
NC='\033[0m'

# Конфигурация проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SCRIPT_VERSION="2.0.0"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
CONTAINER_NAME="sglang-qwen-inference"
IMAGE_NAME="sglang-qwen-inference:latest"

# Режимы развёртывания
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-single-gpu}"

# Функции логирования
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[$timestamp] [INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp] [WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[$timestamp] [ERROR]${NC} $message"
            ;;
        DEBUG)
            echo -e "${BLUE}[$timestamp] [DEBUG]${NC} $message"
            ;;
        CPU)
            echo -e "${CYAN}[$timestamp] [CPU]${NC} $message"
            ;;
        GPU2)
            echo -e "${MAGENTA}[$timestamp] [GPU2]${NC} $message"
            ;;
        OFFLOAD)
            echo -e "${ORANGE}[$timestamp] [OFFLOAD]${NC} $message"
            ;;
    esac
}

# Показать справку
show_help() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  SGLang + Qwen2.5-0.5B-Instruct Docker Launcher v${SCRIPT_VERSION}                      ║${NC}"
    echo -e "${CYAN}║  Поддержка: Single-GPU, Dual-GPU, CPU-only, CPU Offload                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Использование: $0 [команда] [опции]"
    echo ""
    echo "Команды:"
    echo "  build       Сборка Docker образа"
    echo "  start       Запуск контейнера"
    echo "  stop        Остановка контейнера"
    echo "  restart     Перезапуск контейнера"
    echo "  logs        Просмотр логов контейнера"
    echo "  status      Статус контейнера"
    echo "  shell       Вход в контейнер (interactive shell)"
    echo "  test        Тестирование API"
    echo "  cleanup     Очистка ресурсов"
    echo "  gpu-check   Проверка GPU"
    echo "  help        Показать эту справку"
    echo ""
    echo "Режимы развёртывания (через переменную окружения DEPLOYMENT_MODE):"
    echo "  single-gpu  Одиночный GPU (по умолчанию)"
    echo "  dual-gpu    Два GPU (требует 2 GPU)"
    echo "  cpu-only    Только CPU (без GPU)"
    echo "  offload     CPU Offload (частичная загрузка в GPU)"
    echo ""
    echo "Опции:"
    echo "  --no-cache    Не использовать кэш при сборке (для build)"
    echo "  --force       Принудительная пересборка (для build)"
    echo "  --detach      Запуск в фоновом режиме (для start)"
    echo "  --follow      Следовать за логами (для logs)"
    echo ""
    echo "Примеры:"
    echo "  # Одиночный GPU (по умолчанию)"
    echo "  ./scripts/start.sh build && ./scripts/start.sh start"
    echo ""
    echo "  # Dual-GPU режим"
    echo "  DEPLOYMENT_MODE=dual-gpu ./scripts/start.sh build && \\"
    echo "  DEPLOYMENT_MODE=dual-gpu ./scripts/start.sh start"
    echo ""
    echo "  # CPU-only режим"
    echo "  DEPLOYMENT_MODE=cpu-only ./scripts/start.sh build && \\"
    echo "  DEPLOYMENT_MODE=cpu-only ./scripts/start.sh start"
    echo ""
    echo "  # CPU Offload режим"
    echo "  DEPLOYMENT_MODE=offload ./scripts/start.sh build && \\"
    echo "  DEPLOYMENT_MODE=offload ./scripts/start.sh start"
    echo ""
}

# Проверка зависимостей
check_dependencies() {
    log INFO "Проверка зависимостей..."
    
    local missing_deps=()
    
    # Проверка Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    else
        log INFO "Docker найден: $(docker --version)"
    fi
    
    # Проверка Docker Compose (v2)
    if command -v docker compose &> /dev/null; then
        log INFO "Docker Compose найден"
    elif command -v docker-compose &> /dev/null; then
        log INFO "Docker Compose (v1) найден"
    else
        missing_deps+=("docker-compose")
    fi
    
    # Проверка NVIDIA Container Toolkit (только для GPU режимов)
    if [ "$DEPLOYMENT_MODE" != "cpu-only" ]; then
        if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
            log WARN "NVIDIA Container Toolkit не настроен. GPU может быть недоступен."
            log WARN "Установите: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        else
            log INFO "NVIDIA Container Toolkit готов"
        fi
    else
        log CPU "CPU-only режим - NVIDIA Toolkit не требуется"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log ERROR "Отсутствуют зависимости: ${missing_deps[*]}"
        log ERROR "Установите недостающие компоненты перед запуском"
        exit 1
    fi
    
    log INFO "Все зависимости присутствуют"
}

# Проверка GPU
check_gpu() {
    log INFO "Проверка доступности GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        log INFO "Информация о GPU:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu --format=csv
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Подсчёт количества GPU
        local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        log INFO "Обнаружено GPU: $gpu_count"
        
        # Проверка режима
        case $DEPLOYMENT_MODE in
            dual-gpu)
                if [ "$gpu_count" -lt 2 ]; then
                    log ERROR "Для dual-gpu режима требуется минимум 2 GPU. Обнаружено: $gpu_count"
                    return 1
                fi
                log GPU2 "Dual-GPU режим активирован"
                ;;
            single-gpu|offload)
                if [ "$gpu_count" -lt 1 ]; then
                    log WARN "GPU не обнаружен. Переключение в CPU-only режим..."
                    DEPLOYMENT_MODE="cpu-only"
                    return 0
                fi
                log INFO "Single-GPU режим активирован"
                ;;
            cpu-only)
                log CPU "CPU-only режим - GPU не используется"
                ;;
        esac
        
        return 0
    else
        if [ "$DEPLOYMENT_MODE" != "cpu-only" ]; then
            log WARN "nvidia-smi не найден. Переключение в CPU-only режим..."
            DEPLOYMENT_MODE="cpu-only"
        fi
        log CPU "CPU-only режим активен"
        return 0
    fi
}

# Определение Dockerfile по режиму
get_dockerfile() {
    case $DEPLOYMENT_MODE in
        cpu-only)
            echo "Dockerfile.cpu"
            ;;
        dual-gpu)
            echo "Dockerfile.gpu2"
            ;;
        offload)
            echo "Dockerfile.offload"
            ;;
        single-gpu|*)
            echo "Dockerfile"
            ;;
    esac
}

# Определение порта по режиму
get_default_port() {
    case $DEPLOYMENT_MODE in
        cpu-only)
            echo "5000"
            ;;
        dual-gpu)
            echo "5000"
            ;;
        offload)
            echo "5000"
            ;;
        single-gpu|*)
            echo "5000"
            ;;
    esac
}

# Сборка Docker образа
build_image() {
    local dockerfile=$(get_dockerfile)
    local port=$(get_default_port)
    
    log INFO "Сборка Docker образа: $IMAGE_NAME"
    log INFO "Режим: $DEPLOYMENT_MODE"
    log INFO "Dockerfile: $dockerfile"
    log INFO "Порт: $port"
    
    # Проверка существования Dockerfile
    if [ ! -f "$dockerfile" ]; then
        log ERROR "Dockerfile не найден: $dockerfile"
        exit 1
    fi
    
    local build_args="--build-arg HF_TOKEN=${HF_TOKEN:-}"
    build_args="$build_args --build-arg DEPLOYMENT_MODE=$DEPLOYMENT_MODE"
    
    # Проверка флагов
    local no_cache=false
    local force=false
    
    for arg in "$@"; do
        case $arg in
            --no-cache)
                no_cache=true
                ;;
            --force)
                force=true
                ;;
        esac
    done
    
    if [ "$force" = true ]; then
        log WARN "Принудительная пересборка (--force)"
        docker compose build --build-arg "DOCKERFILE=$dockerfile" --no-cache || docker-compose build --build-arg "DOCKERFILE=$dockerfile" --no-cache
    elif [ "$no_cache" = true ]; then
        log INFO "Сборка без кэша (--no-cache)"
        docker compose build --build-arg "DOCKERFILE=$dockerfile" --no-cache || docker-compose build --build-arg "DOCKERFILE=$dockerfile" --no-cache
    else
        docker compose build --build-arg "DOCKERFILE=$dockerfile" || docker-compose build --build-arg "DOCKERFILE=$dockerfile"
    fi
    
    if [ $? -eq 0 ]; then
        log INFO "Docker образ успешно собран: $IMAGE_NAME"
    else
        log ERROR "Ошибка сборки Docker образа"
        exit 1
    fi
}

# Запуск контейнера
start_container() {
    local dockerfile=$(get_dockerfile)
    local port="${SGLANG_PORT:-$(get_default_port)}"
    
    log INFO "Запуск контейнера: $CONTAINER_NAME"
    log INFO "Режим: $DEPLOYMENT_MODE"
    log INFO "Порт: $port"
    
    local detach=false
    for arg in "$@"; do
        case $arg in
            --detach)
                detach=true
                ;;
        esac
    done
    
    # Проверка статуса контейнера
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            log WARN "Контейнер $CONTAINER_NAME уже запущен"
            return 0
        else
            log INFO "Контейнер найден в остановленном состоянии, перезапускаем..."
            docker compose down || docker-compose down
        fi
    fi
    
    # Установка переменных окружения
    export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
    export SGLANG_PORT="${port}"
    export SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
    export SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
    export DEPLOYMENT_MODE="$DEPLOYMENT_MODE"
    export DOCKERFILE="$dockerfile"
    
    # CPU-only специфичные переменные
    if [ "$DEPLOYMENT_MODE" = "cpu-only" ]; then
        export SGLANG_CPU_OMP_NUM_THREADS="${SGLANG_CPU_OMP_NUM_THREADS:-8}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
        log CPU "CPU threads: $SGLANG_CPU_OMP_NUM_THREADS"
    fi
    
    # Dual-GPU специфичные переменные
    if [ "$DEPLOYMENT_MODE" = "dual-gpu" ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
        export GPU_COUNT=2
        log GPU2 "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
    
    # Offload специфичные переменные
    if [ "$DEPLOYMENT_MODE" = "offload" ]; then
        export SGLANG_ENABLE_CPU_OFFLOAD="true"
        export OFFLOAD_DIR="${OFFLOAD_DIR:-/tmp/offload}"
        log OFFLOAD "CPU Offload включён"
    fi
    
    # Исправление прав доступа к папке models на хосте
    log INFO "Исправление прав доступа к папке models..."
    if [ -d "$PROJECT_DIR/models" ]; then
        sudo chmod -R 777 "$PROJECT_DIR/models" 2>/dev/null || log WARN "Не удалось изменить права на models"
    fi
    
    if [ "$detach" = true ]; then
        log INFO "Запуск в фоновом режиме..."
        docker compose up -d || docker-compose up -d
    else
        docker compose up || docker-compose up
    fi
    
    if [ $? -eq 0 ]; then
        log INFO "Контейнер успешно запущен"
        log INFO "API доступен по адресу: http://localhost:$port"
        log INFO "Режим: $DEPLOYMENT_MODE"
    else
        log ERROR "Ошибка запуска контейнера"
        exit 1
    fi
}

# Остановка контейнера
stop_container() {
    log INFO "Остановка контейнера: $CONTAINER_NAME"
    
    docker compose down || docker-compose down
    
    if [ $? -eq 0 ]; then
        log INFO "Контейнер успешно остановлен"
    else
        log WARN "Контейнер не был запущен или уже остановлен"
    fi
}

# Перезапуск контейнера
restart_container() {
    log INFO "Перезапуск контейнера: $CONTAINER_NAME"
    log INFO "Режим: $DEPLOYMENT_MODE"
    stop_container
    sleep 2
    start_container
}

# Просмотр логов
view_logs() {
    local follow=false
    for arg in "$@"; do
        case $arg in
            --follow)
                follow=true
                ;;
        esac done
    
    log INFO "Просмотр логов (режим: $DEPLOYMENT_MODE)..."
    
    if [ "$follow" = true ]; then
        log INFO "Просмотр логов с отслеживанием (Ctrl+C для выхода)..."
        docker compose logs -f || docker-compose logs -f
    else
        docker compose logs || docker-compose logs
    fi
}

# Статус контейнера
check_status() {
    log INFO "Статус контейнера: $CONTAINER_NAME"
    log INFO "Режим: $DEPLOYMENT_MODE"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Информация о GPU использовании (только для GPU режимов)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$" && [ "$DEPLOYMENT_MODE" != "cpu-only" ]; then
        echo ""
        log INFO "Использование GPU контейнером:"
        nvidia-smi
    fi
}

# Вход в контейнер
enter_shell() {
    log INFO "Вход в контейнер: $CONTAINER_NAME"
    
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log ERROR "Контейнер $CONTAINER_NAME не запущен. Запустите его сначала."
        exit 1
    fi
    
    docker exec -it $CONTAINER_NAME /bin/bash || docker exec -it $CONTAINER_NAME /bin/sh
}

# Тестирование API
test_api() {
    local port="${SGLANG_PORT:-$(get_default_port)}"
    log INFO "Тестирование API на порту $port..."
    log INFO "Режим: $DEPLOYMENT_MODE"
    
    # Проверка health endpoint
    echo ""
    echo "Тест 1: Health Check"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if curl -s "http://localhost:$port/health" | grep -q "OK"; then
        log INFO "Health check: OK"
    else
        log WARN "Health check: Не удалось получить ответ (сервер может загружаться)"
    fi
    
    # Тест генерации
    echo ""
    echo "Тест 2: Генерация текста"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local test_response=$(curl -s -X POST "http://localhost:$port/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "prompt": "Расскажи коротко о искусственном интеллекте.",
            "max_tokens": 100,
            "temperature": 0.7
        }')
    
    if echo "$test_response" | grep -q "text\|response"; then
        log INFO "Генерация: OK"
        echo "$test_response" | head -c 500
        echo ""
    else
        log WARN "Генерация: Не удалось получить ответ"
        echo "$test_response"
    fi
    
    echo ""
    echo "Тест 3: Информация о модели"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    curl -s "http://localhost:$port/v1/models" | head -c 500
    echo ""
}

# Очистка ресурсов
cleanup() {
    log INFO "Очистка ресурсов..."
    log INFO "Режим: $DEPLOYMENT_MODE"
    
    # Остановка контейнера
    log INFO "Остановка контейнера..."
    stop_container
    
    # Очистка неиспользуемых образов и томов
    log INFO "Очистка неиспользуемых Docker ресурсов..."
    docker system prune -f
    
    log INFO "Очистка завершена"
}

# Обработка аргументов командной строки
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local command=$1
    shift
    
    # Отображение выбранного режима
    log INFO "Выбранный режим развёртывания: $DEPLOYMENT_MODE"
    
    case $command in
        build)
            check_dependencies
            build_image "$@"
            ;;
        start)
            check_dependencies
            check_gpu
            start_container "$@"
            ;;
        stop)
            stop_container
            ;;
        restart)
            restart_container
            ;;
        logs)
            view_logs "$@"
            ;;
        status)
            check_status
            ;;
        shell)
            enter_shell
            ;;
        test)
            test_api
            ;;
        cleanup)
            cleanup
            ;;
        gpu-check)
            check_gpu
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log ERROR "Неизвестная команда: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
