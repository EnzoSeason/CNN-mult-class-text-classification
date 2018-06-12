import pickle
import sys
from nltk import RegexpTokenizer
import string
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

#python 3
MAX_SEQUENCE_LENGTH = 81

# 1. Create toknizer and stopwords
stop = list(string.punctuation) + ["«", "»", "“", "”","’"]
toknizer = RegexpTokenizer(r'''\'|\w+|[^\w\s]''')

# 2. Load tokenizer
tokenizer_file = open("tokenizer.pickle", "rb")
tokenizer = pickle.load(tokenizer_file)
tokenizer_file.close()

# 3. Load Model
model = load_model('cnn_simple.h5')

# 4. get tweets
tweets = sys.argv[1:]

# 5. prediction
tweets_token = []
for tweet in tweets:
  token = [i for i in toknizer.tokenize(tweet.strip().lower()) if i not in stop]
  text = ''.join(str(e)+' ' for e in token)
  tweets_token.append(text)
sequences = tokenizer.texts_to_sequences(tweets_token)
test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

predicts = model.predict(test)
scores = []
for predict in predicts:
  scores.append(predict.tolist())

# 6. create a recommended list
category_label = [i for i in range(13)]

def getRecommendedList(score, category_label):
  d_raw = dict(zip(score, category_label))
  items = d_raw.items()
  d = sorted(items, reverse=True)
  recommended_list_raw = [i[1] for i in d]
  recommended_list = []
  for i in range(13):
    category_id = recommended_list_raw[i]
    if (recommended_list_raw[i] == 0):
      category_id = 101
    if (recommended_list_raw[i] == 10):
      category_id = 50
    if (recommended_list_raw[i] == 11):
      category_id = 56
    if (recommended_list_raw[i] == 12):
      category_id = 69
    recommended_list.append(category_id)
  print(recommended_list)

for score in scores:
  getRecommendedList(score, category_label)
