# For case if .yaml is not working
conda create -n alignn
conda activate alignn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
# python setup.py develop
conda install -c dglteam/label/cu113 dgl -y
conda install -c conda-forge wandb tensorbowandard -y
pip install alignn
conda install pymatgen jarvis-tools -y
pip install rdkit dgllife