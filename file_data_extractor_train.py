import os
from data_extractor_train import TrainDocument
import csv


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
    features = []

    for ans_filename in os.listdir(ans_directory):
        doc_filename = next(filename for filename in doc_filenames if filename == ans_filename[:-4])
        doc_filepath = os.path.join(docs_directory, doc_filename)
        ans_filepath = os.path.join(ans_directory, ans_filename)
        doc = (TrainDocument(doc_filepath, ans_filepath, get_ml_features=True))
        doc_features = doc.get_feature_vector()
        for feature in doc_features:
            features.append(feature)

    
    with open('output_50d_2.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    return features

if __name__ == '__main__':
    extract_files_train_ml()

    