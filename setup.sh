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

# Find Python command (python3 or python)
find_python_cmd() {
    if command -v python3 &> /dev/null; then
        echo "python3"
        return
    elif command -v python &> /dev/null; then
        # Check if Python version is 3.x
        if python -c "import sys; assert sys.version_info[0]==3" &> /dev/null; then
            echo "python"
            return
        fi
    fi
    log_error "Python 3 is required but not installed."
}

# Check required commands
log_step "Checking prerequisites..."

# Find Python command
PYTHON_CMD=$(find_python_cmd)
log_success "${PYTHON_CMD} found"

# Find pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD="${PYTHON_CMD} -m pip"
fi
log_success "${PIP_CMD} command found"

# Start the application
log_step "Starting the application..."
${PYTHON_CMD} app.py &
app_pid=$!
spinner $app_pid

# Wait for user interrupt
trap "kill $app_pid; echo 'Interrupted, closing app.'; exit" SIGINT SIGTERM

# Wait for app to exit
wait $app_pid
log_success "Application closed."
