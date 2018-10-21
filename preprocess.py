import gzip
import collections
import json

RAW_PATH = "data/raw/reviews_Electronics.json.gz"
PROCESSED_PATH = "data/processed/single_product.json"
TO_PRINT = 10
TO_PROCESS = 100000 #-1 means process all
PICKED_PRODUCT = "B00001WRSJ"

def parse (path):
  g = gzip.open (path, 'r')
  for l in g:
    yield eval (l)


ratings = []
products = collections.defaultdict(list)
products_review_count = collections.defaultdict(int)
count = 0
for review in parse (RAW_PATH):
  products[review['asin']].append(review)
  products_review_count[review['asin']] += 1
  count += 1
  if count % 100000 == 0:
    print ("Processed: " + str(count))

  if count == TO_PROCESS and TO_PROCESS != -1:
    break

p_sorted = sorted(products_review_count.items(), key = lambda item: item[1], 
                  reverse = True)

print ("MOST REVIEWED PRODUCTS")
for idx in range(TO_PRINT):
  key, value = p_sorted [idx]
  print (key + ": " + str (products_review_count [key]))


print ("Saving product " + PICKED_PRODUCT)
with open(PROCESSED_PATH, 'w') as f:
  json.dump(products[PICKED_PRODUCT], f, ensure_ascii=False, indent=2)
