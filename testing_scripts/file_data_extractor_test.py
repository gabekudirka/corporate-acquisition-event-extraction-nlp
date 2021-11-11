import os
from data_extractor_test import TestDocument

def extract_files_test(num_files = -1):
    docs_directory = './data/docs'
    doc_files = os.listdir(docs_directory)
    docs = []

    for docs_filename in doc_files[:num_files]:
        doc_filepath = os.path.join(docs_directory, docs_filename)
        docs.append(TestDocument(doc_filepath))

    return docs