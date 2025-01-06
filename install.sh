sudo apt-get update && sudo apt-get install -y git python3 python3-venv && \ # Authenticates User
if [ ! -d "conductor_venv" ]; then python3 -m venv conductor_venv; fi && \ 

source conductor_venv/bin/activate && \
pip install --upgrade pip gradio openai GitPython && \ # Localizes Dependencies

if [ -d "conductor" ]; then (cd conductor && git pull); else git clone https://github.com/rabbidave/conductor.git; fi && \
cd conductor && python app.py # Localizes Application
