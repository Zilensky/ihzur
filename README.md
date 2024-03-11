# ihzur
project in ihzur group 16

# search_frontend
# all the imports of the file
    def import
    import pickle
    from flask import Flask, request, jsonify
    from inverted_index_gcp import InvertedIndex
    import sys
    from collections import Counter, OrderedDict
    import itertools
    from itertools import islice, count, groupby
    import pandas as pd
    import os
    import re
    from operator import itemgetter
    from time import time
    from pathlib import Path
    import pickle
    from google.cloud import storage
    from collections import defaultdict
    from contextlib import closing
    import nltk
    from nltk.corpus import stopwords
    import math
    import traceback
    from autocorrect import Speller
    from urllib.parse import quote



    class MyFlaskApp(Flask):
        def run(self, host=None, port=None, debug=None, **options):
            super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)
    
    app = MyFlaskApp(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# tokenize function for later use for the query
    def tokenize_and_steem_word(text):
      RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
      return [token.group() for token in RE_WORD.finditer(text.lower())]

# This will be our system to rank the documents, it takes the query implements all the liabraries to make it usebale
# then it takes all the data from out bucket and calculates the nm25 score for each word of the query by using the
# posting_list adds the bm25 scores and sorts it from best to worst and prints the best 100

    @app.route("/search")
    def search():
      # Take all the stopwords
  
      english_stopwords = frozenset(stopwords.words('english'))
      corpus_stopwords = ["category", "references", "also", "external", "links",
                          "may", "first", "see", "history", "people", "one", "two",
                          "part", "thumb", "including", "second", "following",
                          "many", "however", "would", "became"]
      all_stopwords = english_stopwords.union(corpus_stopwords)
      
      # Add stemmer and Speller corrections
      ps = nltk.stem.PorterStemmer()
      res = []
      spell = Speller()
      query = request.args.get('query', '')
      if len(query) == 0:
          return jsonify(res)
          
      # Tokenize and process the query
      query_terms = tokenize_and_steem_word(query)  # Split query into individual terms
      
      # Retrieve the inverted index for titles
      file_path = 'postings_gcp/index.pkl'
      storagegcp = storage.Client()
      bucket = storagegcp.bucket('works89651257')
      passage = bucket.blob(file_path)
      content = passage.download_as_bytes()
      index_title = pickle.loads(content)
      doc_len_use = index_title.doc_len
      
      # Prepare to store BM25 scores for each document
      bm25_scores = defaultdict(float)
      
      # Calculate average document length
      avg_doc_len = sum(doc_len_use.values()) / len(doc_len_use)
      
      # Calculate BM25 scores for each query term in each document
      try:
          for term in query_terms:
              # make sure the term is fine to work with all the crrections
              if term in all_stopwords:
                  continue
              term = spell(term)
              term = ps.stem(term)
              # Retrieve the posting list for the query term
              posting_list = index_title.read_a_posting_list('.', term, 'works89651257')
              # Calculate IDF for the query term
              idf = math.log((len(doc_len_use) - len(posting_list) + 0.5) / (len(posting_list) + 0.5))
              # Update BM25 scores for documents containing the query term using k1 of 1.5 amd b of 0.75
              try:
                  for doc_id, tf in posting_list:
                      if doc_id == 0:
                          continue
                      doc_len = doc_len_use[doc_id]
                      k1 = 1.5
                      b = 0.75
                      bm25 = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len))))
                      bm25_scores[doc_id] += bm25
              except Exception as e:
                  return jsonify(str(traceback.print_exc()), "first")
      except Exception as e:
          return jsonify(str(e), "sirst")
      try:
          # Sort the search results based on BM25 scores
          sorted_results = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
          # Retrieve titles for the sorted documents
          for doc_id, _ in sorted_results:
              res.append((str(doc_id), index_title.idtotitledict[doc_id]))
          return jsonify(res[:100])
      except Exception as e:
          return jsonify(str(e), "tirst")

# Indexing file
project - jupyter 
"At the start we bring all imports and pip install necessary.
after bringing the path, we starrt building the inverted index, we start
by defining functions and variables as followed:

    token2bucket_id(token) - using hashing creates an id for token
    # all the following funcs utilize map reduce thecnice in order to get better performance.

    word_count(text, doc_id)-  regexing and stemming non-stopword words of a doc, returns a posting list after 
    counting words in a doc in the form of Counter() and returning a posting list out of it

    reduce_word_counts(unsorted_pl) - sorts posting lists

    calculate_df(postings) - calculating df by the posting lists

    partition_postings_and_write(postings_filtered) - hashing the PLs and writing the PLs to
    the index instance

    tokenize_and_steem_word(text): tokenizes (regexes) text of words

now we are entering main...
we iterate over the paths and write down to the index path by path to use less memory, we are 
starting the loop by bringing parrquet files, one with id_title for later on update
the inverted index instace with a dict that maps id's to titles for the use of calculations as data.

then we want a dict doc id and how many words it holds so we bring it to the variable
data2, this is used to perform bm25 in the search quary, this one also goes to the II instance.

then, similarly to assigment 3 we perform the steps to prepare variables for the index
creation, in each iteration we updated the index instance with the requierd data and remove it from
the buffer and by that preventing kernl crashes

# inside the for loop
    inverted.df.update(w2df_dict)
    inverted.idtotitledict.update(data)
    inverted.doc_len.update(data2)

after finishing with the for loop we updating posting locs the the instance, also 
in a for loop to save memory and finally creating index.pkl.

all the code is done with good structuring and even estimated time print for each path.

code:

# project group 16
# 212240741 323794131
# data retrieval
    !gcloud dataproc clusters list --region us-central1

    !pip install -q google-cloud-storage==1.43.0
    !pip install -q graphframes

    import pyspark
    import sys
    from collections import Counter, OrderedDict, defaultdict
    import itertools
    from itertools import islice, count, groupby
    import pandas as pd
    import os
    import re
    from operator import itemgetter
    import nltk
    from nltk.stem.porter import *
    from nltk.corpus import stopwords
    from time import time
    from pathlib import Path
    import pickle
    import pandas as pd
    from google.cloud import storage


    import hashlib
    def _hash(s):
        return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

    nltk.download('stopwords')

    !ls -l /usr/lib/spark/jars/graph*
    spark
    # Put your bucket name below and make sure you can access it without an error
    bucket_name = 'works89651257' 
    full_path = f"gs://{bucket_name}/"
    paths=[]

    client = storage.Client()
    blobs = client.list_blobs(bucket_name)
    for b in blobs:      
        if b.name != 'graphframes.sh' and  'docsss' not in b.name and  'postings_gcp' not in b.name and 'postings_new' not in b.name:
            paths.append(full_path+b.name)
  

    %cd -q /home/dataproc
    !ls inverted_index_gcp.py
    import inverted_index_gcp
    from inverted_index_gcp import InvertedIndex
    sc.addFile("/home/dataproc/inverted_index_gcp.py")
    sys.path.insert(0,SparkFiles.getRootDirectory())

### functions and variables to be used ###
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    NUM_BUCKETS = 124
    time
    def token2bucket_id(token):
      return int(_hash(token),16) % NUM_BUCKETS

    def word_count(text, doc_id):
        # applaying stem
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        ps = nltk.stem.PorterStemmer()
        tokens = [ps.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
        word_counts = Counter(tokens)
        posting_list = [(token, (doc_id, tf)) for token, tf in word_counts.items()]
        return posting_list



    def reduce_word_counts(unsorted_pl):
        sorted_pl = sorted(unsorted_pl, key=lambda x: x[1])
        return sorted_pl



    def calculate_df(postings):
        # Flatten the postings RDD and extract terms
        terms = postings.flatMap(lambda x: [x[0]] * len(x[1]))
        df_counts = terms.map(lambda term: (term, 1)).reduceByKey(lambda a, b: a + b)
        return df_counts



    def partition_postings_and_write(postings_filtered):
        posting_grouped = postings_filtered.map(lambda x: (token2bucket_id(x[0]), x)).groupByKey()
        return posting_grouped.map(lambda x: InvertedIndex.write_a_posting_list(x, "postings_new", "bucket-get-started_x-signifier-416215"))


# tokenization
    def tokenize_and_steem_word(text):
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        return [token.group() for token in RE_WORD.finditer(text.lower())]

# all SW
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
    all_stopwords = english_stopwords.union(corpus_stopwords)
# stemmer
    ps = nltk.stem.PorterStemmer()

# index init variables update for calculations

    inverted = InvertedIndex()
    counter = 0
    sum_list_of_avg_times = 350
    len_list_of_avg_times = 1
    postings_filtered_all = 0
    total = 0
    total_time = 0

    for path in paths:
        x = time()
        data = {}
        data2 = {}
        counter += 1
        # title id dict update
        print(f'path {counter}: title id dict update')
        parquetFile = spark.read.parquet(path)
        # timer stuff
        len_parquet = int(parquetFile.count())
        estimated = len_parquet/(sum_list_of_avg_times/len_list_of_avg_times)
        print(f"estimated time for path {counter}: {estimated} Sec")
        doc_id_title = parquetFile.select("id", "title").rdd.map(lambda row: (row.id, row.title))
        data = doc_id_title.map(lambda x: (x[0], x[1])).collectAsMap()
    
# doc_len update
    print(f'path {counter}: doc_len update')
    doc_id_len = parquetFile.select("id", "text").rdd.map(lambda row: (row.id, row.text))
    length_data = doc_id_len.map(lambda x: (x[0], len([word for word in tokenize_and_steem_word(x[1]) if word not in all_stopwords])))
    data2 = length_data.collectAsMap()
    
# Index update 
    print(f'path {counter}: Index update')
    doc_text_pairs = parquetFile.select("text", "id").rdd
      
    
# writing to the index
    print(f'path {counter}: prapering variables...')
    word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    postings_filtered = postings.filter(lambda x: len(x[1])>1)
    w2df = calculate_df(postings_filtered)
    w2df_dict = w2df.collectAsMap()
    if not postings_filtered_all == 0:
        postings_filtered_all = postings_filtered_all.union(postings_filtered).reduceByKey(lambda x, y: x + y)
    else:
        postings_filtered_all = postings_filtered
    '''_ = partition_postings_and_write(postings_filtered).collect()'''
    
    
    
# finally update the index
    print(f'path {counter}: finally update the index...')
    inverted.df.update(w2df_dict)
    inverted.idtotitledict.update(data)
    inverted.doc_len.update(data2)
    print(f'path {counter} completed')
    y = time()
    total_time += (y - x)
    total += len_parquet
    sum_list_of_avg_times += total / total_time
    len_list_of_avg_times += 1
    print(f"Actuall Time: {(y - x)}\n Estimated Time: {estimated}")
    

    
    
    print(f'All paths are completed!')

# collect all posting lists locations into one super-set
    print(f'path {counter}: started writing PLs to index...')

    _ = partition_postings_and_write(postings_filtered_all).collect()
    print(f'path {counter}: collecting all posting lists locations into one super-set...')

    for blob in client.list_blobs(bucket_name, prefix='postings_new'):
        if not blob.name.endswith("pickle"):
            continue
        with blob.open("rb") as f:
            posting_locs = pickle.load(f)
            for k, v in posting_locs.items():
                if k in inverted.posting_locs.keys():
                    inverted.posting_locs[k].extend(v)
                else:
                    inverted.posting_locs[k] = v
                
# update index in bucket
    print(f'path {counter}: updating index in bucket...')
    inverted.write_index('.', 'index')
    index_src = "index.pkl"
    index_dst = f'gs://{bucket_name}/postings_gcp/{index_src}'
    !gsutil cp $index_src $index_dst
    !gsutil ls -lh $index_dst
    print('Index has been created !')
        
    
    
          




