from data_utils import create_review_db
from config import DATA_PATH


if __name__ == "__main__":
    create_review_db(DATA_PATH, 'data/raw/reviews_Electronics.json.gz')

