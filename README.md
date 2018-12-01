# cs221_project

When running on server, before running any code run : source /home/hanozbhathena/project1

Train sentiment model imdb data:
time CUDA_VISIBLE_DEVICES=0 python sentiment_analysis/sentiment_train.py

Train sentiment amazon data:
time CUDA_VISIBLE_DEVICES=0 python sentiment_analysis/sentiment_train.py --dataset=amazon --num_samples=1000000

Make database for data:
**If running on server run python create_database_server.py else run python create_database.py

Make stats plot for data:
python data_stats.py

Running clustering
python main_cluster.py --extractive_model=all --products=all

Run abstractive summarization
python modules.py 

Run abstractive summarization without breakpoints
python modules.py --debug=False


