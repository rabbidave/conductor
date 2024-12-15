#!/bin/bash

# Function for spinner animation
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to log steps with clear formatting and timestamp
log_step() {
    echo -e "\n🚂 $(date '+%H:%M:%S') \033[1;34m$1\033[0m"
}

# Function to log success
log_success() {
    echo -e "✅ $(date '+%H:%M:%S') \033[1;32m$1\033[0m"
}

# Function to log error and exit
log_error() {
    echo -e "❌ $(date '+%H:%M:%S') \033[1;31mError: $1\033[0m"
    exit 1
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is required but not installed."
    fi
}

# Check required commands
log_step "Checking prerequisites..."
check_command python3 && log_success "Python3 found"
check_command git && log_success "Git found"
check_command curl && log_success "Curl found"

# Default values using OpenAI endpoint and models
MODEL_A_URL=${1:-"https://api.openai.com/v1"}
MODEL_B_URL=${2:-"https://api.openai.com/v1"}
MODEL_A_ID=${3:-"gpt-4o-mini"}
MODEL_B_ID=${4:-"gpt-4o-mini"}
MODEL_A_ALIAS=${5:-"model_a"}
MODEL_B_ALIAS=${6:-"model_b"}
MAX_TOKENS=${7:-2000}
TEMPERATURE=${8:-0.7}
TOP_P=${9:-0.95}
DEFAULT_API_KEY="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"

# Create temporary directory
log_step "Creating temporary directory..."
TEMP_DIR=$(mktemp -d) || log_error "Failed to create temporary directory"
cd "$TEMP_DIR" || log_error "Failed to change to temporary directory"
log_success "Created temporary directory: $TEMP_DIR"

# Clone the repository
log_step "Cloning conductor repository..."
git clone https://github.com/rabbidave/conductor.git > /dev/null 2>&1 &
spinner $!
if [ $? -eq 0 ]; then
    log_success "Repository cloned successfully"
else
    log_error "Failed to clone repository"
fi
cd conductor || log_error "Failed to change to conductor directory"

# Create .env file
log_step "Creating .env file..."
cat > .env << EOF
MODEL_A_URL="$MODEL_A_URL"
MODEL_B_URL="$MODEL_B_URL"
MODEL_A_ID="$MODEL_A_ID"
MODEL_B_ID="$MODEL_B_ID"
MODEL_A_ALIAS="$MODEL_A_ALIAS"
MODEL_B_ALIAS="$MODEL_B_ALIAS"
MAX_TOKENS="$MAX_TOKENS"
TEMPERATURE="$TEMPERATURE"
TOP_P="$TOP_P"
OPENAI_API_KEY="$DEFAULT_API_KEY"
EOF
log_success "Environment configuration created"

# Set up Python virtual environment and install dependencies
log_step "Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate || . .venv/bin/activate || log_error "Failed to activate virtual environment"
log_success "Virtual environment activated"

log_step "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install gradio openai GitPython > /dev/null 2>&1 &
spinner $!
if [ $? -eq 0 ]; then
    log_success "Dependencies installed successfully"
else
    log_error "Failed to install dependencies"
fi

# Launch the application
log_step "Launching Conductor..."
echo -e "\033[1;33mConductor will start on http://localhost:31337\033[0m"
python app.py

# Cleanup on script exit
trap 'log_step "Cleaning up..."; rm -rf "$TEMP_DIR"; log_success "Cleanup complete"' EXIT
