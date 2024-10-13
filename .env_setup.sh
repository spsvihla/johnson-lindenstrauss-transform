# set up vscode environment
python3 -m venv .venv 
source .venv/bin/activate
pip install numpy
pip install .
echo "PYTHONPATH=.venv/lib/python3.10/site-packages/" > .env

# required to run genomics_example.ipynb
pip install ipykernel scipy matplotlib
python3 -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
