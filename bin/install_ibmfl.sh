#!/usr/bin/env bash
echo "Creating tf1 conda environment..."
conda create -n tf1 python=3.6 tensorflow=1.15 -y
conda activate tf1
echo "Created tf1 environment. Installing IBM FL..."
cd /mydata/ && git clone https://github.com/IBM/federated-learning-lib
cd /mydata/federated-learning-lib/federated-learning-lib && pip install ./federated_learning_lib-1.0.4-py3-none-any.whl
echo "Installed IBM FL. Installing Jupyter Notebook..."
conda install -c conda-forge notebook -y
jupyter notebook --generate-config
echo "c.NotebookApp.allow_origin = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
ufw allow 8888
echo "Installed Jupyter Notebook"

echo "Creating tf2 conda environment..."
conda create -n tf2 python=3.6 -y
conda activate tf2
echo "Created tf2 environment. Installing IBM FL..."
cd /mydata/federated-learning-lib/federated-learning-lib && pip install ./federated_learning_lib-1.0.4-py3-none-any.whl
echo "Installed IBM FL. Installing Tensorflow..."
conda install tensorflow -y
echo "Installing Jupyter Notebook..."
conda install -c conda-forge notebook -y
echo "Installed Jupyter Notebook"

echo "Copying notebook and DP Implementation..."
cp /local/repository/etc/keras_fl_model.py /mydata/conda/envs/tf2/lib/python3.6/site-packages/ibmfl/model/keras_fl_model.py
cp /local/repository/etc/keras_classifier_aggregator.ipynb /mydata/federated-learning-lib/Notebooks/keras_classifier/keras_classifier_aggregator.ipynb
cp /local/repository/etc/keras_classifier_p0.ipynb /mydata/federated-learning-lib/Notebooks/keras_classifier/keras_classifier_p0.ipynb
cp /local/repository/etc/keras_classifier_p1.ipynb /mydata/federated-learning-lib/Notebooks/keras_classifier/keras_classifier_p1.ipynb
echo "Done."
