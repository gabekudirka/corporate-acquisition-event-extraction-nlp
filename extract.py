from data_extractor_test import TestDocument
import os
import sys


def extract_files_test(num_files = -1):
    docs_directory = './data/docs'
    doc_files = os.listdir(docs_directory)
    docs = []

    for docs_filename in doc_files[:num_files]:
        doc_filepath = os.path.join(docs_directory, docs_filename)
        docs.append(TestDocument(doc_filepath))

    return docs

extracted_files = extract_files_test()


# test_files = open(sys.argv[1], "r").readlines()
# test_files = [file.strip() for file in test_files]    

# extract_info = []
# for filepath in test_files:
#     extract_info.append(TestDocument(filepath))

#template_file = sys.argv[1] + '.template'
template_file = 'test_doclist.template'
for template in extracted_files:
    template.print_to_file(template_file)



#print('hi')