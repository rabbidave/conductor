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
if ! command -v pip3 &> /dev/null ; then
    check_command python && log_success "python command found"
    pip_command="python3 -m pip"
else
    pip_command="pip3"
    log_success "pip3 command found"
fi

# Start the application
log_step "Starting the application..."
python3 app.py &
app_pid=$!
spinner $app_pid

# Wait for user interrupt
trap "kill $app_pid; echo 'Interrupted, closing app.'; exit" SIGINT SIGTERM

# Wait for app to exit
wait $app_pid
log_success "Application closed."
