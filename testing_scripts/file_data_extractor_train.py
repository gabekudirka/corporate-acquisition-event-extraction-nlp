import os
from data_extractor_train import TrainDocument


def extract_files_train():
    docs_directory = './data/docs'
    ans_directory = './data/anskeys'
    doc_filenames = os.listdir(docs_directory)
    docs = []

    for ans_filename in os.listdir(ans_directory):
        doc_filename = next(filename for filename in doc_filenames if filename == ans_filename[:-4])
        doc_filepath = os.path.join(docs_directory, doc_filename)
        ans_filepath = os.path.join(ans_directory, ans_filename)
        docs.append(TrainDocument(doc_filepath, ans_filepath))

    return docs

def extract_files_train_ml():
    docs_directory = './data/docs'
    ans_directory = './data/anskeys'
    doc_filenames = os.listdir(docs_directory)
    docs = []

    for ans_filename in os.listdir(ans_directory):
        doc_filename = next(filename for filename in doc_filenames if filename == ans_filename[:-4])
        doc_filepath = os.path.join(docs_directory, doc_filename)
        ans_filepath = os.path.join(ans_directory, ans_filename)
        docs.append(TrainDocument(doc_filepath, ans_filepath, get_ml_features=True))

    return docs