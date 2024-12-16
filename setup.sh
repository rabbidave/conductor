#!/bin/bash

# Set the new defaults as environment variables
export MODEL_A_URL="https://openrouter.ai/api/v1"
export MODEL_A_ID="google/gemini-2.0-flash-exp:free"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Start the Python script in the background
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux"* ]]; then # macOS or Linux
  .venv/bin/python app.py &
elif [[ "$OSTYPE" == "msys"*  || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"*  ]]; then # Windows (WSL, Cygwin, etc)
  .venv/Scripts/python.exe app.py &
else
    echo "Unable to determine OS python location."
    exit 1
fi

# Wait for the server to start
sleep 5  # You might need to adjust this based on how long it takes for your server to start up

# Construct the URL and open the default browser to it
SERVER_URL="http://0.0.0.0:31337" 

if [[ "$OSTYPE" == "darwin"* ]]; then # macOS
  open "$SERVER_URL"
elif [[ "$OSTYPE" == "linux"* ]]; then # Linux
  xdg-open "$SERVER_URL"
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* ]]; then # Windows (WSL, Cygwin, etc)
  explorer "$SERVER_URL"
else
    echo "Unable to determine OS for opening browser."
fi
