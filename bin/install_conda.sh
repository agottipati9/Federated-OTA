#!/usr/bin/env bash

echo "Installing conda..."
wget -P /mydata https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
cd /mydata && bash ./Miniconda2-latest-Linux-x86_64.sh -b -p /mydata/conda
eval "$(/mydata/conda/bin/conda shell.bash hook)"
/mydata/conda/bin/conda init
echo "Installed conda in /mydata/conda. Please restart the shell."
