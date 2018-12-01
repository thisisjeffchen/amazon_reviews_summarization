from data_utils import create_review_db
#from config import DATA_PATH
import os
import ipdb as pdb

RAW_DATA_FNAME= os.environ.get('RAW_DATA_FNAME', None)
DATA_DIR= os.environ.get('DATA_PATH', None)


if __name__ == "__main__":
    create_review_db(DATA_DIR, RAW_DATA_FNAME)
