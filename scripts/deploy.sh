#!/usr/bin/env bash
#
# TORCH-RaR Deployment Script
# Supports: vLLM (Docker), OpenRouter (Cloud), LiteLLM Proxy (Docker)
#
# Usage:
#   ./scripts/deploy.sh                    # Interactive menu
#   ./scripts/deploy.sh setup [provider]   # Full setup
#   ./scripts/deploy.sh start [provider]   # Start services
#   ./scripts/deploy.sh stop               # Stop services
#   ./scripts/deploy.sh status             # Show status
#   ./scripts/deploy.sh logs [service]     # View logs
#   ./scripts/deploy.sh test               # Test LLM connection
#   ./scripts/deploy.sh help               # Show help

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Service ports (from docker-compose.yml)
VLLM_PORT=8000
LITELLM_PORT=4000

# Default values
DEFAULT_VLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  TORCH-RaR Deployment Tool${NC}"
    echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════════════${NC}\n"
}

print_section() {
    echo -e "\n${BOLD}${CYAN}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_command() {
    command -v "$1" &> /dev/null
}

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local response

    if [[ "$default" == "y" ]]; then
        read -rp "$prompt [Y/n]: " response
        response="${response:-y}"
    else
        read -rp "$prompt [y/N]: " response
        response="${response:-n}"
    fi

    [[ "${response,,}" == "y" || "${response,,}" == "yes" ]]
}

prompt_value() {
    local prompt="$1"
    local default="${2:-}"
    local response

    if [[ -n "$default" ]]; then
        read -rp "$prompt [$default]: " response
        echo "${response:-$default}"
    else
        read -rp "$prompt: " response
        echo "$response"
    fi
}

# =============================================================================
# Check Functions
# =============================================================================

check_uv() {
    if check_command uv; then
        print_success "uv is installed: $(uv --version)"
        return 0
    else
        print_warning "uv is not installed"
        return 1
    fi
}

install_uv() {
    print_section "Installing uv package manager"

    if check_uv; then
        print_info "uv is already installed"
        return 0
    fi

    print_info "Installing uv via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    if check_uv; then
        print_success "uv installed successfully"
        return 0
    else
        print_error "Failed to install uv"
        return 1
    fi
}

check_docker() {
    print_section "Checking Docker availability"

    if ! check_command docker; then
        print_error "Docker is not installed"
        print_info "Install Docker from: https://docs.docker.com/get-docker/"
        return 1
    fi
    print_success "Docker is installed"

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        print_info "Start Docker with: sudo systemctl start docker"
        return 1
    fi
    print_success "Docker daemon is running"

    # Check Docker Compose
    if docker compose version &> /dev/null; then
        print_success "Docker Compose (plugin) is available"
    elif check_command docker-compose; then
        print_success "Docker Compose (standalone) is available"
    else
        print_error "Docker Compose is not installed"
        return 1
    fi

    return 0
}

check_nvidia() {
    print_section "Checking NVIDIA GPU availability"

    if ! check_command nvidia-smi; then
        print_warning "nvidia-smi not found - GPU support may not be available"
        return 1
    fi

    if nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
            print_info "  $line"
        done
        return 0
    else
        print_warning "NVIDIA drivers may not be properly installed"
        return 1
    fi
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_env() {
    print_section "Setting up environment"

    local env_file="$PROJECT_ROOT/.env"
    local env_example="$PROJECT_ROOT/.env.example"

    if [[ -f "$env_file" ]]; then
        print_info ".env file already exists"
        if prompt_yes_no "Do you want to recreate it?"; then
            cp "$env_file" "$env_file.backup.$(date +%Y%m%d%H%M%S)"
            print_info "Backup created"
        else
            return 0
        fi
    fi

    if [[ ! -f "$env_example" ]]; then
        print_error ".env.example not found"
        return 1
    fi

    cp "$env_example" "$env_file"
    print_success "Created .env from template"

    # Interactive configuration
    echo ""
    print_info "Let's configure your environment:"
    echo ""

    # OpenRouter API Key
    local openrouter_key
    openrouter_key=$(prompt_value "Enter your OpenRouter API key (or press Enter to skip)")
    if [[ -n "$openrouter_key" ]]; then
        sed -i "s/OPENROUTER_API_KEY=.*/OPENROUTER_API_KEY=$openrouter_key/" "$env_file"
        print_success "OpenRouter API key configured"
    fi

    # vLLM Model
    local vllm_model
    vllm_model=$(prompt_value "vLLM model name" "$DEFAULT_VLLM_MODEL")
    sed -i "s|VLLM_MODEL=.*|VLLM_MODEL=$vllm_model|" "$env_file"
    print_success "vLLM model configured: $vllm_model"

    # HuggingFace Token (for gated models)
    if prompt_yes_no "Do you need HuggingFace token for gated models?"; then
        local hf_token
        hf_token=$(prompt_value "Enter your HuggingFace token")
        if [[ -n "$hf_token" ]]; then
            sed -i "s/HUGGING_FACE_HUB_TOKEN=.*/HUGGING_FACE_HUB_TOKEN=$hf_token/" "$env_file"
            print_success "HuggingFace token configured"
        fi
    fi

    print_success "Environment configuration complete"
}

# =============================================================================
# Dependency Installation
# =============================================================================

install_deps() {
    print_section "Installing Python dependencies"

    cd "$PROJECT_ROOT"

    if ! check_uv; then
        install_uv
    fi

    print_info "Running uv sync..."
    uv sync

    print_success "Dependencies installed successfully"
}

# =============================================================================
# Config Validation
# =============================================================================

validate_config_files() {
    print_section "Validating configuration files"

    local has_errors=false

    # Check settings.yaml
    if [[ -f "$PROJECT_ROOT/settings.yaml" ]]; then
        if python3 -c "import yaml; yaml.safe_load(open('$PROJECT_ROOT/settings.yaml'))" 2>/dev/null; then
            print_success "settings.yaml is valid YAML"
        else
            print_error "settings.yaml has invalid YAML syntax"
            has_errors=true
        fi
    else
        print_error "settings.yaml not found"
        has_errors=true
    fi

    # Check litellm_config.yaml
    if [[ -f "$PROJECT_ROOT/litellm_config.yaml" ]]; then
        if python3 -c "import yaml; yaml.safe_load(open('$PROJECT_ROOT/litellm_config.yaml'))" 2>/dev/null; then
            print_success "litellm_config.yaml is valid YAML"
        else
            print_error "litellm_config.yaml has invalid YAML syntax"
            has_errors=true
        fi
    else
        print_error "litellm_config.yaml not found"
        has_errors=true
    fi

    # Check docker-compose.yml
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        if docker compose -f "$PROJECT_ROOT/docker-compose.yml" config --quiet 2>/dev/null; then
            print_success "docker-compose.yml is valid"
        else
            print_warning "docker-compose.yml may have issues (Docker needed for full validation)"
        fi
    else
        print_error "docker-compose.yml not found"
        has_errors=true
    fi

    # Check .env
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        print_success ".env file exists"

        # Check for required variables
        if grep -q "OPENROUTER_API_KEY=your_openrouter" "$PROJECT_ROOT/.env"; then
            print_warning "OPENROUTER_API_KEY is not configured"
        fi
    else
        print_warning ".env file not found - run setup first"
    fi

    if [[ "$has_errors" == true ]]; then
        return 1
    fi

    print_success "All configuration files validated"
    return 0
}

# =============================================================================
# API Testing
# =============================================================================

test_vllm_health() {
    print_info "Testing vLLM health endpoint..."

    if curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        print_success "vLLM server is healthy"
        return 0
    else
        print_error "vLLM server is not responding on port $VLLM_PORT"
        return 1
    fi
}

test_litellm_health() {
    print_info "Testing LiteLLM proxy health endpoint..."

    if curl -sf "http://localhost:$LITELLM_PORT/health" > /dev/null 2>&1; then
        print_success "LiteLLM proxy is healthy"
        return 0
    else
        print_error "LiteLLM proxy is not responding on port $LITELLM_PORT"
        return 1
    fi
}

test_openrouter() {
    print_info "Testing OpenRouter API connectivity..."

    # Load API key from .env
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    if [[ -z "${OPENROUTER_API_KEY:-}" || "$OPENROUTER_API_KEY" == "your_openrouter_api_key_here" ]]; then
        print_error "OPENROUTER_API_KEY not configured"
        return 1
    fi

    local response
    response=$(curl -sf -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        "https://openrouter.ai/api/v1/models" 2>&1) || {
        print_error "Failed to connect to OpenRouter API"
        return 1
    }

    print_success "OpenRouter API is accessible"
    return 0
}

test_llm_connection() {
    print_section "Testing LLM Connection"

    cd "$PROJECT_ROOT"

    # Load environment
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    print_info "Running TORCH-RaR test command..."

    if uv run python main.py test -v; then
        print_success "LLM connection test passed"
        return 0
    else
        print_error "LLM connection test failed"
        return 1
    fi
}

# =============================================================================
# Service Management
# =============================================================================

get_docker_compose_cmd() {
    if docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

start_vllm() {
    print_section "Starting vLLM Service"

    cd "$PROJECT_ROOT"

    # Load environment
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    # Check for NVIDIA GPU
    if ! check_nvidia; then
        print_error "vLLM requires NVIDIA GPU with CUDA support"
        return 1
    fi

    # Check if already running
    if docker ps --format '{{.Names}}' | grep -q "vllm"; then
        print_info "vLLM container is already running"
        return 0
    fi

    print_info "Starting vLLM container..."
    print_info "This may take several minutes for first run (downloading model)"

    local compose_cmd
    compose_cmd=$(get_docker_compose_cmd)

    $compose_cmd up -d vllm

    print_info "Waiting for vLLM to be ready (checking health)..."
    local attempts=0
    local max_attempts=60  # 5 minutes with 5-second intervals

    while [[ $attempts -lt $max_attempts ]]; do
        if test_vllm_health 2>/dev/null; then
            echo ""
            print_success "vLLM service started successfully"
            return 0
        fi
        sleep 5
        ((attempts++))
        echo -ne "\r${BLUE}[INFO]${NC} Waiting... ($attempts/$max_attempts)"
    done

    echo ""
    print_error "vLLM failed to start within timeout"
    print_info "Check logs with: ./scripts/deploy.sh logs vllm"
    return 1
}

start_litellm_proxy() {
    print_section "Starting LiteLLM Proxy"

    cd "$PROJECT_ROOT"

    # Load environment
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    # LiteLLM depends on vLLM in the with-proxy profile
    print_info "Starting vLLM and LiteLLM services..."

    local compose_cmd
    compose_cmd=$(get_docker_compose_cmd)

    $compose_cmd --profile with-proxy up -d

    print_info "Waiting for services to be ready..."
    local attempts=0
    local max_attempts=60

    while [[ $attempts -lt $max_attempts ]]; do
        if test_litellm_health 2>/dev/null; then
            echo ""
            print_success "LiteLLM proxy started successfully"
            return 0
        fi
        sleep 5
        ((attempts++))
        echo -ne "\r${BLUE}[INFO]${NC} Waiting... ($attempts/$max_attempts)"
    done

    echo ""
    print_error "LiteLLM proxy failed to start within timeout"
    print_info "Check logs with: ./scripts/deploy.sh logs litellm"
    return 1
}

stop_services() {
    print_section "Stopping Services"

    cd "$PROJECT_ROOT"

    local compose_cmd
    compose_cmd=$(get_docker_compose_cmd)

    print_info "Stopping all Docker services..."
    $compose_cmd --profile with-proxy down

    print_success "All services stopped"
}

service_status() {
    print_section "Service Status"

    cd "$PROJECT_ROOT"

    echo -e "\n${BOLD}Docker Containers:${NC}"
    if docker ps --filter "name=vllm" --filter "name=litellm" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -v "^NAMES" | grep -q .; then
        docker ps --filter "name=vllm" --filter "name=litellm" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        print_info "No TORCH-RaR containers running"
    fi

    echo -e "\n${BOLD}Port Status:${NC}"

    # vLLM
    if curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        print_success "vLLM (port $VLLM_PORT): HEALTHY"
    elif nc -z localhost $VLLM_PORT 2>/dev/null; then
        print_warning "vLLM (port $VLLM_PORT): LISTENING (not healthy yet)"
    else
        print_info "vLLM (port $VLLM_PORT): NOT LISTENING"
    fi

    # LiteLLM
    if curl -sf "http://localhost:$LITELLM_PORT/health" > /dev/null 2>&1; then
        print_success "LiteLLM Proxy (port $LITELLM_PORT): HEALTHY"
    elif nc -z localhost $LITELLM_PORT 2>/dev/null; then
        print_warning "LiteLLM Proxy (port $LITELLM_PORT): LISTENING (not healthy yet)"
    else
        print_info "LiteLLM Proxy (port $LITELLM_PORT): NOT LISTENING"
    fi

    echo -e "\n${BOLD}Current Configuration:${NC}"
    if [[ -f "$PROJECT_ROOT/settings.yaml" ]]; then
        local provider
        provider=$(grep "^llm_provider:" "$PROJECT_ROOT/settings.yaml" | awk '{print $2}')
        print_info "LLM Provider: $provider"
    fi
}

view_logs() {
    local service="${1:-}"

    print_section "Viewing Logs"

    cd "$PROJECT_ROOT"

    local compose_cmd
    compose_cmd=$(get_docker_compose_cmd)

    case "$service" in
        vllm)
            $compose_cmd logs -f vllm
            ;;
        litellm|proxy)
            $compose_cmd logs -f litellm-proxy
            ;;
        *)
            print_info "Showing logs for all services (Ctrl+C to exit)"
            $compose_cmd --profile with-proxy logs -f
            ;;
    esac
}

# =============================================================================
# Provider Setup Functions
# =============================================================================

setup_vllm() {
    print_section "Setting up vLLM Provider"

    # Check requirements
    if ! check_docker; then
        return 1
    fi

    if ! check_nvidia; then
        print_error "vLLM requires NVIDIA GPU"
        return 1
    fi

    # Update settings.yaml
    print_info "Configuring settings.yaml for vLLM..."
    sed -i 's/^llm_provider:.*/llm_provider: vllm/' "$PROJECT_ROOT/settings.yaml"

    print_success "vLLM provider configured"

    if prompt_yes_no "Start vLLM service now?" "y"; then
        start_vllm
    fi
}

setup_openrouter() {
    print_section "Setting up OpenRouter Provider"

    # Load existing .env
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    if [[ -z "${OPENROUTER_API_KEY:-}" || "$OPENROUTER_API_KEY" == "your_openrouter_api_key_here" ]]; then
        print_warning "OpenRouter API key not configured"
        local api_key
        api_key=$(prompt_value "Enter your OpenRouter API key")
        if [[ -n "$api_key" ]]; then
            if [[ -f "$PROJECT_ROOT/.env" ]]; then
                sed -i "s/OPENROUTER_API_KEY=.*/OPENROUTER_API_KEY=$api_key/" "$PROJECT_ROOT/.env"
            else
                echo "OPENROUTER_API_KEY=$api_key" > "$PROJECT_ROOT/.env"
            fi
            export OPENROUTER_API_KEY="$api_key"
            print_success "API key saved to .env"
        else
            print_error "API key is required for OpenRouter"
            return 1
        fi
    fi

    # Update settings.yaml
    print_info "Configuring settings.yaml for OpenRouter..."
    sed -i 's/^llm_provider:.*/llm_provider: openrouter/' "$PROJECT_ROOT/settings.yaml"

    print_success "OpenRouter provider configured"

    # Test connection
    if prompt_yes_no "Test OpenRouter connection now?" "y"; then
        test_openrouter
    fi
}

setup_litellm_proxy() {
    print_section "Setting up LiteLLM Proxy Provider"

    if ! check_docker; then
        return 1
    fi

    # Update settings.yaml
    print_info "Configuring settings.yaml for LiteLLM Proxy..."
    sed -i 's/^llm_provider:.*/llm_provider: litellm_proxy/' "$PROJECT_ROOT/settings.yaml"

    print_success "LiteLLM Proxy provider configured"

    if prompt_yes_no "Start LiteLLM Proxy now?" "y"; then
        start_litellm_proxy
    fi
}

# =============================================================================
# Full Setup
# =============================================================================

full_setup() {
    local provider="${1:-}"

    print_header
    print_section "Full Setup"

    # Step 1: Install uv
    if ! check_uv; then
        if prompt_yes_no "Install uv package manager?" "y"; then
            install_uv
        else
            print_error "uv is required"
            return 1
        fi
    fi

    # Step 2: Setup environment
    setup_env

    # Step 3: Install dependencies
    install_deps

    # Step 4: Validate config
    validate_config_files

    # Step 5: Provider selection
    if [[ -z "$provider" ]]; then
        echo ""
        print_info "Select your LLM provider:"
        echo ""
        echo "  1) vLLM        - Local GPU inference (requires NVIDIA GPU + Docker)"
        echo "  2) OpenRouter  - Cloud API (no Docker needed, requires API key)"
        echo "  3) LiteLLM     - Docker proxy with routing (requires Docker)"
        echo ""

        local choice
        read -rp "Enter choice [1-3]: " choice

        case "$choice" in
            1) provider="vllm" ;;
            2) provider="openrouter" ;;
            3) provider="litellm_proxy" ;;
            *)
                print_error "Invalid choice"
                return 1
                ;;
        esac
    fi

    # Step 6: Provider-specific setup
    case "$provider" in
        vllm)
            setup_vllm
            ;;
        openrouter)
            setup_openrouter
            ;;
        litellm|litellm_proxy)
            setup_litellm_proxy
            ;;
        *)
            print_error "Unknown provider: $provider"
            return 1
            ;;
    esac

    # Step 7: Final test
    if prompt_yes_no "Run LLM connection test?" "y"; then
        test_llm_connection
    fi

    print_section "Setup Complete"
    print_success "TORCH-RaR is ready to use!"
    echo ""
    print_info "Run the pipeline with: uv run python main.py run --limit 10"
}

# =============================================================================
# Interactive Menu
# =============================================================================

interactive_menu() {
    print_header

    while true; do
        echo ""
        echo -e "${BOLD}Main Menu${NC}"
        echo "─────────────────────────────────"
        echo "  1) Full Setup"
        echo "  2) Start Services"
        echo "  3) Stop Services"
        echo "  4) Service Status"
        echo "  5) View Logs"
        echo "  6) Test LLM Connection"
        echo "  7) Validate Configuration"
        echo "  8) Change Provider"
        echo "  9) Help"
        echo "  0) Exit"
        echo ""

        local choice
        read -rp "Enter choice [0-9]: " choice

        case "$choice" in
            1) full_setup ;;
            2)
                echo ""
                echo "Select provider to start:"
                echo "  1) vLLM"
                echo "  2) LiteLLM Proxy (includes vLLM)"
                read -rp "Enter choice [1-2]: " subchoice
                case "$subchoice" in
                    1) start_vllm ;;
                    2) start_litellm_proxy ;;
                esac
                ;;
            3) stop_services ;;
            4) service_status ;;
            5)
                echo ""
                echo "Select service logs:"
                echo "  1) vLLM"
                echo "  2) LiteLLM Proxy"
                echo "  3) All"
                read -rp "Enter choice [1-3]: " subchoice
                case "$subchoice" in
                    1) view_logs vllm ;;
                    2) view_logs litellm ;;
                    3) view_logs ;;
                esac
                ;;
            6) test_llm_connection ;;
            7) validate_config_files ;;
            8)
                echo ""
                echo "Select new provider:"
                echo "  1) vLLM"
                echo "  2) OpenRouter"
                echo "  3) LiteLLM Proxy"
                read -rp "Enter choice [1-3]: " subchoice
                case "$subchoice" in
                    1) setup_vllm ;;
                    2) setup_openrouter ;;
                    3) setup_litellm_proxy ;;
                esac
                ;;
            9) show_help ;;
            0)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                ;;
        esac
    done
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << 'EOF'

TORCH-RaR Deployment Tool
=========================

Usage:
  ./scripts/deploy.sh                    Interactive menu (default)
  ./scripts/deploy.sh setup [provider]   Full setup with optional provider
  ./scripts/deploy.sh start [service]    Start services
  ./scripts/deploy.sh stop               Stop all services
  ./scripts/deploy.sh status             Show service status
  ./scripts/deploy.sh logs [service]     View service logs
  ./scripts/deploy.sh test               Test LLM connection
  ./scripts/deploy.sh validate           Validate config files
  ./scripts/deploy.sh help               Show this help

Providers:
  vllm          Local GPU inference via Docker (requires NVIDIA GPU)
  openrouter    Cloud API (requires API key, no Docker needed)
  litellm_proxy Docker proxy with routing to multiple backends

Services:
  vllm          vLLM container (port 8000)
  litellm       LiteLLM proxy container (port 4000)

Examples:
  ./scripts/deploy.sh setup openrouter   Setup with OpenRouter
  ./scripts/deploy.sh setup vllm         Setup with local vLLM
  ./scripts/deploy.sh start vllm         Start vLLM container
  ./scripts/deploy.sh logs vllm          View vLLM logs
  ./scripts/deploy.sh test               Test current configuration

For more information, see the README.md file.

EOF
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    local command="${1:-}"

    case "$command" in
        setup)
            full_setup "${2:-}"
            ;;
        start)
            case "${2:-}" in
                vllm)
                    start_vllm
                    ;;
                litellm|proxy|litellm_proxy)
                    start_litellm_proxy
                    ;;
                "")
                    print_error "Usage: $0 start [vllm|litellm]"
                    exit 1
                    ;;
                *)
                    print_error "Unknown service: ${2:-}"
                    print_info "Available services: vllm, litellm"
                    exit 1
                    ;;
            esac
            ;;
        stop)
            stop_services
            ;;
        status)
            service_status
            ;;
        logs)
            view_logs "${2:-}"
            ;;
        test)
            test_llm_connection
            ;;
        validate)
            validate_config_files
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            interactive_menu
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
