import sys
from data_extractor_test import TestDocument

#Extract filepaths from input doclist
test_files = open(sys.argv[1], "r").readlines()
test_files = [file.strip() for file in test_files]    

#Extract data from each document in the doc list
extracted_docs = []
for filepath in test_files:
    extracted_docs.append(TestDocument(filepath))

#Output to file
template_file = sys.argv[1] + '.template'
for template in extracted_docs:
    template.print_to_file(template_file)
