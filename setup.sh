# install requirements.txt for project
# Usage: source setup.sh

# create virtual environment
python3 -m venv venv_NLP_exam

# activate virtual environment
source ./venv_NLP_exam/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Install mistral 7B model and saves it in 'model' folder
huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir model --local-dir-use-symlinks False

# run the code
python3 main_mistral.py

# Extra: Type ```python3 main_mistral.py --help``` to see the help message for the adjustable parameters

# Deactivate virtual environment
deactivate