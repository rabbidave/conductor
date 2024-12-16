#!/bin/bash
set -e

echo "🚂 Conductor Installation Script"
echo "-------------------------------"

# Detect OS
OS="unknown"
case "$(uname -s)" in
    Linux*)     OS="linux";;
    Darwin*)    OS="mac";;
    MINGW*|MSYS*|CYGWIN*) OS="windows";;
esac

# Function to check if running with admin/sudo privileges
check_privileges() {
    if [ "$OS" = "windows" ]; then
        # Check if running as Administrator in Windows
        net session >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Error: Please run this script with administrator privileges"
            echo "Right-click on PowerShell and select 'Run as administrator'"
            exit 1
        fi
    else
        # Check if running with sudo on Unix
        if [ "$EUID" -ne 0 ]; then
            echo "Error: Please run this script with sudo"
            exit 1
        fi
    fi
}

# Windows-specific installation
install_windows() {
    echo "Detected Windows OS"
    
    # Download and run PowerShell script
    cat > install.ps1 << 'EOF'
$ErrorActionPreference = "Stop"
Write-Host "Installing Python for Windows..."

# Check if Python is already installed
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    Write-Host "Python is already installed at: $($pythonCommand.Source)"
    Write-Host "Python version: $(python --version)"
} else {
    # Download and install Python
    Write-Host "Downloading Python installer..."
    $installerUrl = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"
    
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    Write-Host "Installing Python..."
    Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    Remove-Item $installerPath
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Clone the repository if git is available
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Cloning repository..."
    git clone https://github.com/yourusername/conductor.git
    Set-Location conductor
} else {
    Write-Host "Git not found. Please install git or download the repository manually."
}

Write-Host "Setup completed! You can now run app.py"
EOF

    powershell.exe -ExecutionPolicy Bypass -File install.ps1
}

# Linux installation
install_linux() {
    echo "Detected Linux OS"
    
    if command -v apt-get &>/dev/null; then
        # Debian/Ubuntu
        apt-get update
        apt-get install -y python3 python3-pip git
    elif command -v yum &>/dev/null; then
        # RHEL/CentOS
        yum update -y
        yum install -y python3 python3-pip git
    else
        echo "Unsupported Linux distribution"
        exit 1
    fi
    
    # Clone repository
    git clone https://github.com/yourusername/conductor.git
    cd conductor
}

# macOS installation
install_mac() {
    echo "Detected macOS"
    
    # Install Homebrew if not present
    if ! command -v brew &>/dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Python and Git
    brew install python git
    
    # Clone repository
    git clone https://github.com/yourusername/conductor.git
    cd conductor
}

# Main installation logic
main() {
    check_privileges
    
    case "$OS" in
        "windows")
            install_windows
            ;;
        "linux")
            install_linux
            ;;
        "mac")
            install_mac
            ;;
        *)
            echo "Unsupported operating system"
            exit 1
            ;;
    esac
    
    echo "✅ Installation completed!"
    echo "You can now run: python app.py"
}

# Run main installation
main
