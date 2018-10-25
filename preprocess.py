import gzip
import collections
import json

RAW_PATH = "data/raw/reviews_Electronics.json.gz"
PROCESSED_PATH = "data/processed/"
TO_PRINT = 25
TO_PROCESS = -1 #-1 means process all
PICKED_PRODUCTS = ["B00001WRSJ", "B009AYLDSU", "B007I5JT4S", "B000EI0EB8", "B0007OWASE", "B00008OE43"]
REVIEW_HUMAN_COUNT = 75

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

idx_75 = 0
for idx in range (len (p_sorted)):
  if (p_sorted[idx][1] <= REVIEW_HUMAN_COUNT):
    idx_75 = idx
    break

print ("PRODUCTS WITH BETWEEN 75-100 REVIEWS")
for idx in range(TO_PRINT):
  key, value = p_sorted [idx_75 + idx]
  print (key + ": " + str (products_review_count [key]))

for asin in PICKED_PRODUCTS:
  print ("Saving product " + asin)
  with open(PROCESSED_PATH + asin + ".json", 'w') as f:
   json.dump(products[asin], f, ensure_ascii=False, indent=2)
