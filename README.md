# Recognizing themes in Amazon reviews through Unsupervised Multi-Document Summarization

## Installing

```
conda create --name cs221_project python=3.6
source activate cs221_project
pip install -r requirements.txt
```

Unzip data
```
unzip data.zip
```

## Running clustering
```
mkdir tmp; 
export TFHUB_CACHE_DIR=tmp; python main_cluster.py
```
