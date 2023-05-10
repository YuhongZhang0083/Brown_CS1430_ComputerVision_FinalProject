conda env remove -n "csci1470"
conda env create -n "csci1470" -f env_setup/csci1470.yml

## Install new environment.
python3 -m ipykernel install --user --name csci1470 --display-name "DL-S23 (3.10)"
