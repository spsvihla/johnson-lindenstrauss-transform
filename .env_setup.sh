python3 -m venv .venv 
source .venv/bin/activate
pip install numpy
pip install .
echo "PYTHONPATH=.venv/lib/python3.10/site-packages/" > .env
pip install ipykernel scipy matplotlib
python3 -m ipykernel install --user --name=.venv --display-name "Python (.venv)"