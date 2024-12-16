#!/bin/bash

# Enable error handling
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
log() {
    echo -e "${GREEN}[+]${NC} $1"
}

error() {
    echo -e "${RED}[!]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[*]${NC} $1"
}

# Function to check Python version
check_python() {
    if command -v python3 &>/dev/null; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log "Found Python version: $python_version"
        return 0
    else
        warn "Python 3 not found"
        return 1
    fi
}

# Function to install Python based on OS
install_python() {
    case "$(uname -s)" in
        Linux*)
            if command -v apt-get &>/dev/null; then
                log "Using apt to install Python..."
                sudo apt-get update
                sudo apt-get install -y python3 python3-pip
            elif command -v yum &>/dev/null; then
                log "Using yum to install Python..."
                sudo yum update -y
                sudo yum install -y python3 python3-pip
            else
                error "Unsupported Linux distribution"
                exit 1
            fi
            ;;
            
        Darwin*)
            if command -v brew &>/dev/null; then
                log "Using Homebrew to install Python..."
                brew install python
            else
                log "Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                brew install python
            fi
            ;;
            
        MINGW*|MSYS*|CYGWIN*)
            error "For Windows, please install Python 3.11 from python.org"
            error "Download URL: https://www.python.org/downloads/"
            exit 1
            ;;
            
        *)
            error "Unsupported operating system"
            exit 1
            ;;
    esac
}

# Function to set environment variables
setup_env() {
    log "Setting up environment variables..."
    # Using printf to ensure proper escaping of special characters
    printf 'export MODEL_A_URL="https://openrouter.ai/api/v1"\n' >> ~/.bashrc
    printf 'export MODEL_A_ID="google/gemini-2.0-flash-exp:free"\n' >> ~/.bashrc
    source ~/.bashrc
}

# Main installation process
main() {
    log "Starting Conductor setup..."
    
    # Check if Python is installed
    if ! check_python; then
        log "Installing Python..."
        install_python
    fi
    
    # Verify Python installation
    if ! check_python; then
        error "Python installation failed"
        exit 1
    fi
    
    # Set up environment variables
    setup_env
    
    log "Setup completed successfully!"
    log "You can now run: python app.py"
}

# Run main installation
main
