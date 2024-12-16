#!/bin/bash

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    OS='macos' ;;
        Linux*)     OS='linux' ;;
        MINGW*|MSYS*|CYGWIN*) OS='windows' ;;
        *)          OS='unknown' ;;
    esac
    echo "Detected OS: $OS"
}

# Install Python 3 based on OS
install_python() {
    case "$OS" in
        "macos")
            if command -v brew &> /dev/null; then
                brew install python
            else
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                brew install python
            fi
            ;;
        "linux")
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y python3
            elif command -v yum &> /dev/null; then
                sudo yum update
                sudo yum install -y python3
            elif command -v pacman &> /dev/null; then
                sudo pacman -Sy python
            else
                echo "Unable to determine package manager. Please install Python 3 manually."
                exit 1
            fi
            ;;
        "windows")
            echo "Downloading Python installer..."
            curl -o python-installer.exe https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe
            ./python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
            rm python-installer.exe
            ;;
        *)
            echo "Unsupported operating system"
            exit 1
            ;;
    esac
}

# Check and install Python if needed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing..."
    detect_os
    install_python
fi

# Set the new defaults as environment variables
export MODEL_A_URL="https://openrouter.ai/api/v1"
export MODEL_A_ID="google/gemini-2.0-flash-exp:free"

# Run the application (which will handle venv setup)
python3 app.py
