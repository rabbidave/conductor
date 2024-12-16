#!/bin/bash

# Set the new defaults as environment variables
export MODEL_A_URL="https://openrouter.ai/api/v1"
export MODEL_A_ID="google/gemini-2.0-flash-exp:free"

# Start the Python script in the background
python app.py &

# Wait for the server to start
sleep 5  # You might need to adjust this based on how long it takes for your server to start up

# Construct the URL and open the default browser to it
SERVER_URL="http://0.0.0.0:31337" 


if [[ "$OSTYPE" == "darwin"* ]]; then # macOS
  open "$SERVER_URL"
elif [[ "$OSTYPE" == "linux"* ]]; then # Linux
  xdg-open "$SERVER_URL"
elif [[ "$OSTYPE" == "msys"*  || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"*  ]]; then # Windows (WSL, Cygwin, etc)
  explorer "$SERVER_URL"
else
    echo "Unable to determine OS for opening browser."
fi
