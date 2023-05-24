# For case if .yaml is not working
conda create -n alignn-segnn
conda activae alignn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
# python setup.py develop
conda install -c dglteam/label/cu113 dgl -y
conda install -c conda-forge wandb tensorbowandard -y
pip install alignn -y
conda install pymatgen jarvis-tools -y
pip install rdkit dgllife
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install e3nn