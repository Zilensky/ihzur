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





