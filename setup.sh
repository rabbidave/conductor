#!/bin/bash

# Function to log steps with clear formatting
log_step() {
    echo -e "\n🚂 \033[1;34m$1\033[0m"
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ Error: $1 is required but not installed."
        exit 1
    fi
}

# Check required commands
log_step "Checking prerequisites..."
check_command python3
check_command git
check_command curl

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
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone the repository
log_step "Cloning conductor repository..."
git clone https://github.com/rabbidave/conductor.git
cd conductor

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

# Launch the application
log_step "Launching Conductor..."
python3 app.py

# Cleanup on script exit
trap 'rm -rf "$TEMP_DIR"' EXIT
