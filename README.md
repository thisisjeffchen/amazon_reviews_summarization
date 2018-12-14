# Recognizing themes in Amazon reviews through Unsupervised Multi-Document Summarization

## Installing

```
conda create --name cs221_project python=3.6
source activate cs221_project
pip install -r requirements.txt
```

Unzip data (very small subset)
```
unzip data.zip
```

## Running extractive 
```
mkdir tmp; 
export TFHUB_CACHE_DIR=tmp; 
python main_cluster.py --prepare_embeddings=True --embeddings_preprocessed=False;
python main_cluster.py --prepare_embeddings=False --embeddings_preprocessed=True;
```

## Running abstractive
```
python model_data.py
python main_abs.py --train_abs=True --debug=False --test_abs=False --cold_start=True
python main_abs.py --train_abs=False --debug=False --test_abs=True --cold_start=False
```
