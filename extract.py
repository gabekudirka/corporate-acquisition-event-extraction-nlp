from data_extractor import TrainDocument, TestDocument
import os
import sys


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

def extract_files_test(num_files = -1):
    docs_directory = './data/docs'
    doc_files = os.listdir(docs_directory)
    docs = []

    for docs_filename in doc_files[:num_files]:
        doc_filepath = os.path.join(docs_directory, docs_filename)
        docs.append(TestDocument(doc_filepath))

    return docs

#test_info_extraction = extract_files_test(30)


test_files = open(sys.argv[1], "r").readlines()
test_files = [file.strip() for file in test_files]

extract_info = []
for filepath in test_files:
    test_files.append(TestDocument(filepath))





print('yo')