# install requirements.txt for project
# Usage: source setup.sh

# create virtual environment
python3 -m venv venv_NLP_exam

# activate virtual environment
source ./venv_NLP_exam/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# run the code
python3 main_mistral.py

# Extra: Type ```python3 main_mistral.py --help``` to see the help message for the adjustable parameters

# Deactivate virtual environment
deactivate