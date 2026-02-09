#!/bin/bash
# Скрипт запуска SGLang Docker контейнера с Qwen2.5-0.5B-Instruct
# Оптимизировано для GPU Quadro RTX 4000 (Turing)

set -e

# Цветовые коды для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Конфигурация проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SCRIPT_VERSION="1.0.0"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
CONTAINER_NAME="sglang-qwen-inference"
IMAGE_NAME="sglang-qwen-inference:latest"

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
    esac
}

# Показать справку
show_help() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  SGLang + Qwen2.5-0.5B-Instruct Docker Launcher v${SCRIPT_VERSION}            ║${NC}"
    echo -e "${CYAN}║  Оптимизировано для Quadro RTX 4000 (Turing)                    ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
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
    echo "Опции:"
    echo "  --no-cache    Не использовать кэш при сборке (для build)"
    echo "  --force       Принудительная пересборка (для build)"
    echo "  --detach      Запуск в фоновом режиме (для start)"
    echo "  --follow      Следовать за логами (для logs)"
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
    
    # Проверка NVIDIA Container Toolkit
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        log WARN "NVIDIA Container Toolkit не настроен. GPU может быть недоступен."
        log WARN "Установите: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    else
        log INFO "NVIDIA Container Toolkit готов"
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
        
        # Проверка совместимости с Turing
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
        if echo "$gpu_name" | grep -qi "RTX 4000\|Quadro RTX 4000"; then
            log INFO "Совместимый GPU обнаружен: $gpu_name"
            return 0
        else
            log WARN "Обнаружен GPU: $gpu_name (ожидался Quadro RTX 4000)"
            log WARN "Убедитесь, что ваш GPU поддерживает CUDA и имеет достаточно VRAM"
            return 1
        fi
    else
        log ERROR "nvidia-smi не найден. NVIDIA драйверы не установлены."
        log ERROR "Установите драйверы с https://www.nvidia.com/Download/index.aspx"
        return 1
    fi
}

# Сборка Docker образа
build_image() {
    log INFO "Сборка Docker образа: $IMAGE_NAME"
    
    local build_args="--build-arg HF_TOKEN=${HF_TOKEN:-}"
    
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
        docker compose build --no-cache || docker-compose build --no-cache
    elif [ "$no_cache" = true ]; then
        log INFO "Сборка без кэша (--no-cache)"
        docker compose build --no-cache || docker-compose build --no-cache
    else
        docker compose build || docker-compose build
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
    log INFO "Запуск контейнера: $CONTAINER_NAME"
    
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
    export SGLANG_PORT="${SGLANG_PORT:-5000}"
    export SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
    export SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
    
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
        log INFO "API доступен по адресу: http://localhost:$SGLANG_PORT"
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
        esac
    done
    
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
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Информация о GPU использовании
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
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
    local port="${SGLANG_PORT:-5000}"
    log INFO "Тестирование API на порту $port..."
    
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
