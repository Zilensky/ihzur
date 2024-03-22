# Information Retrieval Engine
In this project we developed an information retrieval engine focused on analyzing all Wikipedia corpus.
This engine aims to identify the most pertinent Wikipedia documents based on a given query.
During the preprocessing phase, we leveraged Google Cloud Platform (GCP) for storage and computing power and utilized various Python libraries including ntlk, pandas and others.

# Project files:
* Indexing file.ipynb: This notebook is utilized to extract data from Wikipedia and construct an inverted index.
* inverted_index_gcp.py: used for creating and using the Inverted index object.
* search_frontend.py: is a Python script responsible for processing user queries, selecting the appropriate search method, and using the Inverted index utilizing the BM25 algorithm to calculate and return the top 100 results.
