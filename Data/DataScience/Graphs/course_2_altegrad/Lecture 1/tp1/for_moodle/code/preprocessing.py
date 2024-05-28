import re
import json
import operator
import itertools
from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import TweetTokenizer

path_read = "./data/"
path_write = "./data/"

min_freq = 5  # retain the words appearing at least this number of times
oov_token = 0  # for out-of-vocabulary words

verbosity = 5

# ========== read and clean reviews ==========

with open(path_read + 'imdb_reviews.txt', 'r', encoding='utf-8') as file:
    reviews = file.readlines()

tokenizer = TweetTokenizer()

cleaned_reviews = []

for counter, rev in enumerate(reviews):
    rev = rev.lower()
    temp = BeautifulSoup(rev, 'lxml')
    text = temp.get_text()  # remove HTML formatting
    text = re.sub(' +', ' ', text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space
    tokens = tokenizer.tokenize(text)  # tokenize
    cleaned_reviews.append(tokens)
    if counter % round(len(reviews)/verbosity) == 0:
        print(counter, '/', len(reviews), 'reviews cleaned')

# ========== build vocab ==========

### fill the gap (create a list 'tokens' containing all the tokens in 'cleaned_reviews') ###
tokens = list(itertools.chain(*cleaned_reviews))

counts = dict(Counter(tokens))

### fill the gap (filter the dictionary 'counts' by retaining only the words that appear at least 'min_freq' times)
counts = {k: v for (k, v) in counts.items() if v >= min_freq}

with open(path_write + 'counts.json', 'w') as file:
    json.dump(counts, file, sort_keys=True, indent=4)

print('counts saved to disk')

sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

# assign to each word an index based on its frequency in the corpus
# the most frequent word will get index equal to 1
# 0 is reserved for out-of-vocabulary words
word_to_index = dict([(my_tuple[0], idx) for idx,my_tuple in enumerate(sorted_counts,1)])

# examples
word_to_index['the']
word_to_index["don't"]

with open(path_write + 'vocab.json', 'w') as file:
    json.dump(word_to_index, file, sort_keys=True, indent=4)

print('vocab saved to disk')

# ========== transform each review into a list of word indexes ==========


reviews_ints = []

for i, rev in enumerate(cleaned_reviews):
    sublist = []
    ### fill the gaps (for the tokens that are not in 'word_to_index', use 'oov_token') ###
    for w in rev:
        if(w in word_to_index.keys()):
            sublist.append(word_to_index[w])
        else:
            sublist.append(oov_token)
    reviews_ints.append(sublist)

with open(path_write + 'doc_ints.txt', 'w') as file:
    for rev in reviews_ints:
        file.write(' '.join([str(elt) for elt in rev]) + '\n')

print('reviews saved to disk')
