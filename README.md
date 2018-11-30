# cs221_project


Train sentiment model imdb data:
time CUDA_VISIBLE_DEVICES=0 python sentiment_train.py

Train sentiment amazon data:
time CUDA_VISIBLE_DEVICES=0 python sentiment_train.py

Make database for data:
python create_database.py

Make stats plot for data:
python data_stats.py

Running clustering
python main_cluster.py --extractive_model=all --products=all
