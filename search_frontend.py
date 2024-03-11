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

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search_title")
def search_title():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    search_title()
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


def tokenize_and_steem_word(text):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    return [token.group() for token in RE_WORD.finditer(text.lower())]


@app.route("/search")
def search():
    #Take all the stopwords
  
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links", 
                "may", "first", "see", "history", "people", "one", "two", 
                "part", "thumb", "including", "second", "following", 
                "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    #Add stemmer and Speller corrections
    ps = nltk.stem.PorterStemmer()
    res = []
    spell = Speller()
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # Tokenize and process the query
    query_terms = tokenize_and_steem_word(query) # Split query into individual terms
    # Retrieve the inverted index for titles
    file_path = 'postings_gcp/index.pkl'
    storagegcp = storage.Client()
    bucket = storagegcp.bucket('bucket-get-started_x-signifier-416215')
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
            #make sure the term is fine to work with with all the crrections
            if term in all_stopwords:
              continue          
            term = spell(term)
            term = ps.stem(term)
            # Retrieve the posting list for the query term
            posting_list = index_title.read_a_posting_list('.', term, 'bucket-get-started_x-signifier-416215')
            # Calculate IDF for the query term
            idf = math.log((len(doc_len_use) - len(posting_list) + 0.5) / (len(posting_list) + 0.5))
            # Update BM25 scores for documents containing the query term
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
                return jsonify(str(traceback.print_exc()),"first")
    except Exception as e:
          return jsonify(str(e),"sirst")
    try:
        # Sort the search results based on BM25 scores
        sorted_results = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        # Retrieve titles for the sorted documents
        for doc_id, _ in sorted_results:
            res.append((str(doc_id), index_title.idtotitledict[doc_id]))
        return jsonify(res[:100])
    except Exception as e:
          return jsonify(str(e),"tirst")



@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':

    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


