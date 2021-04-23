python --version
pip install azure-cli==2.0.72
pip install --upgrade azureml-sdk
pip install azureml-sdk[notebooks]
pip install --upgrade azureml-sdk[cli]
pip install tensorflow==2.0.0
pip install --upgrade "azureml-train-core<0.1.1" "azureml-train-automl<0.1.1" "azureml-contrib-dataset<0.1.1"  --extra-index-url "https://azuremlsdktestpypi.azureedge.net/automl_for_images_private_preview/"